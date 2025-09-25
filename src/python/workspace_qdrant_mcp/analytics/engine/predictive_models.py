"""
Predictive Models Module for forecasting and system optimization.

Provides advanced predictive analytics capabilities including:
- Time series forecasting
- Performance prediction
- Resource utilization forecasting
- System optimization recommendations
- Capacity planning predictions
- Anomaly prediction
"""

import math
import statistics
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

import numpy as np


logger = logging.getLogger(__name__)


class ForecastMethod(Enum):
    """Enumeration for forecasting methods."""
    LINEAR_REGRESSION = "linear_regression"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    MOVING_AVERAGE = "moving_average"
    ARIMA_SIMPLE = "arima_simple"
    SEASONAL_NAIVE = "seasonal_naive"
    TREND_PROJECTION = "trend_projection"


class PredictionHorizon(Enum):
    """Enumeration for prediction time horizons."""
    SHORT_TERM = "short_term"  # 1-10 periods ahead
    MEDIUM_TERM = "medium_term"  # 11-50 periods ahead
    LONG_TERM = "long_term"  # 51+ periods ahead


@dataclass
class ForecastResult:
    """Container for forecasting results."""

    predictions: List[float]
    confidence_intervals: List[Tuple[float, float]]
    method_used: ForecastMethod
    forecast_horizon: int
    accuracy_metrics: Dict[str, float]
    model_parameters: Dict[str, Any]
    prediction_dates: Optional[List[datetime]] = None
    seasonal_components: Optional[List[float]] = None
    trend_component: Optional[List[float]] = None
    residuals: Optional[List[float]] = None
    r_squared: Optional[float] = None
    mae: Optional[float] = None  # Mean Absolute Error
    mape: Optional[float] = None  # Mean Absolute Percentage Error
    is_reliable: bool = True
    reliability_score: float = 0.0
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'predictions': self.predictions,
            'confidence_intervals': self.confidence_intervals,
            'method_used': self.method_used.value,
            'forecast_horizon': self.forecast_horizon,
            'accuracy_metrics': self.accuracy_metrics,
            'model_parameters': self.model_parameters,
            'prediction_dates': [d.isoformat() if d else None for d in (self.prediction_dates or [])],
            'seasonal_components': self.seasonal_components,
            'trend_component': self.trend_component,
            'residuals': self.residuals,
            'r_squared': self.r_squared,
            'mae': self.mae,
            'mape': self.mape,
            'is_reliable': self.is_reliable,
            'reliability_score': self.reliability_score,
            'error_message': self.error_message
        }


@dataclass
class CapacityPrediction:
    """Container for capacity planning predictions."""

    current_utilization: float
    predicted_utilization: List[float]
    capacity_exhaustion_date: Optional[datetime]
    recommended_scaling_points: List[Tuple[datetime, float]]
    growth_rate: float
    confidence_level: float
    resource_type: str
    time_horizon: int
    scaling_recommendations: List[str]
    cost_projections: Optional[Dict[str, float]] = None
    risk_assessment: str = "low"
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'current_utilization': self.current_utilization,
            'predicted_utilization': self.predicted_utilization,
            'capacity_exhaustion_date': self.capacity_exhaustion_date.isoformat() if self.capacity_exhaustion_date else None,
            'recommended_scaling_points': [(d.isoformat(), v) for d, v in self.recommended_scaling_points],
            'growth_rate': self.growth_rate,
            'confidence_level': self.confidence_level,
            'resource_type': self.resource_type,
            'time_horizon': self.time_horizon,
            'scaling_recommendations': self.scaling_recommendations,
            'cost_projections': self.cost_projections,
            'risk_assessment': self.risk_assessment,
            'error_message': self.error_message
        }


@dataclass
class PerformancePrediction:
    """Container for performance predictions."""

    metric_name: str
    current_performance: float
    predicted_performance: List[float]
    performance_trend: str  # "improving", "degrading", "stable"
    bottleneck_predictions: List[str]
    optimization_recommendations: List[str]
    confidence_scores: List[float]
    time_horizon: int
    baseline_comparison: Dict[str, float]
    sla_compliance_prediction: Optional[float] = None
    risk_factors: List[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'metric_name': self.metric_name,
            'current_performance': self.current_performance,
            'predicted_performance': self.predicted_performance,
            'performance_trend': self.performance_trend,
            'bottleneck_predictions': self.bottleneck_predictions,
            'optimization_recommendations': self.optimization_recommendations,
            'confidence_scores': self.confidence_scores,
            'time_horizon': self.time_horizon,
            'baseline_comparison': self.baseline_comparison,
            'sla_compliance_prediction': self.sla_compliance_prediction,
            'risk_factors': self.risk_factors or [],
            'error_message': self.error_message
        }


class PredictiveModels:
    """
    Advanced predictive modeling engine for forecasting and optimization.

    Handles concept drift, insufficient data, and provides robust predictions
    with confidence intervals and reliability assessments.
    """

    def __init__(self, min_training_points: int = 10, confidence_level: float = 0.95):
        """
        Initialize predictive models engine.

        Args:
            min_training_points: Minimum data points required for modeling
            confidence_level: Confidence level for prediction intervals
        """
        self.min_training_points = min_training_points
        self.confidence_level = confidence_level
        self._model_cache = {}
        self._cache_ttl = timedelta(minutes=15)
        self._cache_timestamps = {}

    def forecast_time_series(self,
                           data: List[Union[int, float]],
                           forecast_horizon: int,
                           method: Optional[ForecastMethod] = None,
                           seasonal_period: Optional[int] = None,
                           timestamps: Optional[List[datetime]] = None,
                           cache_key: Optional[str] = None) -> ForecastResult:
        """
        Forecast future values of a time series.

        Args:
            data: Historical time series data
            forecast_horizon: Number of periods to forecast ahead
            method: Forecasting method to use (None = auto-select)
            seasonal_period: Seasonal period if known
            timestamps: Optional timestamps for each data point
            cache_key: Optional cache key for result caching

        Returns:
            ForecastResult with predictions and confidence intervals
        """
        # Check cache if key provided
        if cache_key and self._is_cached_valid(cache_key):
            return self._model_cache[cache_key]

        result = ForecastResult(
            predictions=[],
            confidence_intervals=[],
            method_used=method or ForecastMethod.LINEAR_REGRESSION,
            forecast_horizon=forecast_horizon,
            accuracy_metrics={},
            model_parameters={}
        )

        try:
            # Validate input data
            if not data:
                result.error_message = "Empty dataset provided"
                result.is_reliable = False
                return result

            if len(data) < self.min_training_points:
                result.error_message = f"Insufficient data points (minimum {self.min_training_points} required)"
                result.is_reliable = False
                return result

            if forecast_horizon <= 0:
                result.error_message = "Forecast horizon must be positive"
                result.is_reliable = False
                return result

            # Filter and clean data
            valid_data = []
            valid_indices = []
            valid_timestamps = []

            for i, value in enumerate(data):
                if isinstance(value, (int, float)) and math.isfinite(value):
                    valid_data.append(float(value))
                    valid_indices.append(i)
                    if timestamps and i < len(timestamps):
                        valid_timestamps.append(timestamps[i])

            if len(valid_data) < self.min_training_points:
                result.error_message = "Insufficient valid data points after filtering"
                result.is_reliable = False
                return result

            # Auto-select method if not specified
            if method is None:
                method = self._select_best_method(valid_data, seasonal_period)
                result.method_used = method

            # Generate forecast based on selected method
            if method == ForecastMethod.LINEAR_REGRESSION:
                result = self._forecast_linear_regression(valid_data, forecast_horizon, result)
            elif method == ForecastMethod.EXPONENTIAL_SMOOTHING:
                result = self._forecast_exponential_smoothing(valid_data, forecast_horizon, result)
            elif method == ForecastMethod.MOVING_AVERAGE:
                result = self._forecast_moving_average(valid_data, forecast_horizon, result)
            elif method == ForecastMethod.SEASONAL_NAIVE:
                result = self._forecast_seasonal_naive(valid_data, forecast_horizon, seasonal_period, result)
            elif method == ForecastMethod.TREND_PROJECTION:
                result = self._forecast_trend_projection(valid_data, forecast_horizon, result)
            else:
                result.error_message = f"Unsupported forecasting method: {method}"
                result.is_reliable = False
                return result

            # Calculate accuracy metrics on training data
            result.accuracy_metrics = self._calculate_accuracy_metrics(valid_data, method, seasonal_period)

            # Assess reliability
            result.reliability_score = self._assess_forecast_reliability(valid_data, result)
            result.is_reliable = result.reliability_score > 0.6

            # Generate prediction timestamps if original timestamps provided
            if valid_timestamps:
                result.prediction_dates = self._generate_forecast_timestamps(
                    valid_timestamps, forecast_horizon)

            # Cache result if key provided
            if cache_key:
                self._cache_result(cache_key, result)

        except Exception as e:
            logger.error(f"Error in time series forecasting: {e}")
            result.error_message = f"Forecasting error: {str(e)}"
            result.is_reliable = False

        return result

    def predict_capacity_needs(self,
                             utilization_data: List[Union[int, float]],
                             resource_type: str,
                             time_horizon_days: int = 30,
                             capacity_threshold: float = 0.8) -> CapacityPrediction:
        """
        Predict future capacity needs and scaling requirements.

        Args:
            utilization_data: Historical utilization data (0.0 to 1.0)
            resource_type: Type of resource (e.g., "memory", "cpu", "storage")
            time_horizon_days: Prediction horizon in days
            capacity_threshold: Threshold for capacity warnings

        Returns:
            CapacityPrediction with scaling recommendations
        """
        result = CapacityPrediction(
            current_utilization=0.0,
            predicted_utilization=[],
            capacity_exhaustion_date=None,
            recommended_scaling_points=[],
            growth_rate=0.0,
            confidence_level=0.0,
            resource_type=resource_type,
            time_horizon=time_horizon_days,
            scaling_recommendations=[]
        )

        try:
            if not utilization_data:
                result.error_message = "Empty utilization data provided"
                return result

            # Filter valid data
            valid_data = []
            for value in utilization_data:
                if isinstance(value, (int, float)) and math.isfinite(value):
                    valid_data.append(max(0.0, min(1.0, float(value))))  # Clamp to [0, 1]

            if len(valid_data) < self.min_training_points:
                result.error_message = "Insufficient valid utilization data"
                return result

            result.current_utilization = valid_data[-1]

            # Forecast utilization
            forecast_result = self.forecast_time_series(
                valid_data,
                forecast_horizon=time_horizon_days,
                method=ForecastMethod.LINEAR_REGRESSION
            )

            if not forecast_result.is_reliable:
                result.error_message = "Unable to generate reliable capacity forecast"
                return result

            # Clamp predictions to valid utilization range [0, 1]
            result.predicted_utilization = [max(0.0, min(1.0, pred)) for pred in forecast_result.predictions]
            result.confidence_level = forecast_result.reliability_score

            # Calculate growth rate
            if len(valid_data) > 1:
                periods = len(valid_data) - 1
                result.growth_rate = (valid_data[-1] - valid_data[0]) / periods
            else:
                result.growth_rate = 0.0

            # Find capacity exhaustion point
            base_date = datetime.now()
            for i, util in enumerate(result.predicted_utilization):
                if util >= capacity_threshold:
                    result.capacity_exhaustion_date = base_date + timedelta(days=i + 1)
                    break

            # Generate scaling recommendations
            result.scaling_recommendations = self._generate_scaling_recommendations(
                result, capacity_threshold)

            # Identify recommended scaling points
            for i, util in enumerate(result.predicted_utilization):
                if util >= capacity_threshold * 0.9:  # 90% of threshold
                    scaling_date = base_date + timedelta(days=i + 1)
                    result.recommended_scaling_points.append((scaling_date, util))

            # Assess risk
            max_predicted_util = max(result.predicted_utilization) if result.predicted_utilization else 0
            if max_predicted_util >= 0.95:
                result.risk_assessment = "high"
            elif max_predicted_util >= 0.8:
                result.risk_assessment = "medium"
            else:
                result.risk_assessment = "low"

        except Exception as e:
            logger.error(f"Error in capacity prediction: {e}")
            result.error_message = f"Capacity prediction error: {str(e)}"

        return result

    def predict_performance(self,
                          performance_data: List[Union[int, float]],
                          metric_name: str,
                          time_horizon: int = 30,
                          baseline_value: Optional[float] = None) -> PerformancePrediction:
        """
        Predict future performance metrics and identify optimization opportunities.

        Args:
            performance_data: Historical performance data
            metric_name: Name of the performance metric
            time_horizon: Prediction horizon in periods
            baseline_value: Baseline value for comparison

        Returns:
            PerformancePrediction with optimization recommendations
        """
        result = PerformancePrediction(
            metric_name=metric_name,
            current_performance=0.0,
            predicted_performance=[],
            performance_trend="stable",
            bottleneck_predictions=[],
            optimization_recommendations=[],
            confidence_scores=[],
            time_horizon=time_horizon,
            baseline_comparison={}
        )

        try:
            if not performance_data:
                result.error_message = "Empty performance data provided"
                return result

            # Filter valid data
            valid_data = []
            for value in performance_data:
                if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
                    valid_data.append(float(value))

            if len(valid_data) < self.min_training_points:
                result.error_message = "Insufficient valid performance data"
                return result

            result.current_performance = valid_data[-1]

            # Forecast performance
            forecast_result = self.forecast_time_series(
                valid_data,
                forecast_horizon=time_horizon,
                method=ForecastMethod.EXPONENTIAL_SMOOTHING
            )

            if not forecast_result.is_reliable:
                result.error_message = "Unable to generate reliable performance forecast"
                return result

            result.predicted_performance = forecast_result.predictions
            result.confidence_scores = [forecast_result.reliability_score] * len(forecast_result.predictions)

            # Determine performance trend
            if len(valid_data) > 5:
                recent_avg = statistics.mean(valid_data[-5:])
                earlier_avg = statistics.mean(valid_data[-10:-5]) if len(valid_data) >= 10 else statistics.mean(valid_data[:-5])

                trend_ratio = recent_avg / earlier_avg if earlier_avg > 0 else 1.0

                if trend_ratio > 1.05:  # 5% improvement
                    result.performance_trend = "improving"
                elif trend_ratio < 0.95:  # 5% degradation
                    result.performance_trend = "degrading"
                else:
                    result.performance_trend = "stable"

            # Generate baseline comparison
            if baseline_value:
                result.baseline_comparison = {
                    'current_vs_baseline': (result.current_performance / baseline_value) if baseline_value > 0 else 0,
                    'predicted_vs_baseline': [pred / baseline_value for pred in result.predicted_performance] if baseline_value > 0 else [0] * len(result.predicted_performance)
                }

            # Predict bottlenecks
            result.bottleneck_predictions = self._predict_bottlenecks(valid_data, result.predicted_performance, metric_name)

            # Generate optimization recommendations
            result.optimization_recommendations = self._generate_optimization_recommendations(
                result, valid_data)

            # Assess SLA compliance if applicable
            if baseline_value:
                result.sla_compliance_prediction = self._predict_sla_compliance(
                    result.predicted_performance, baseline_value)

            # Identify risk factors
            result.risk_factors = self._identify_performance_risks(valid_data, result.predicted_performance)

        except Exception as e:
            logger.error(f"Error in performance prediction: {e}")
            result.error_message = f"Performance prediction error: {str(e)}"

        return result

    def detect_concept_drift(self,
                           recent_data: List[Union[int, float]],
                           historical_data: List[Union[int, float]],
                           significance_level: float = 0.05) -> Dict[str, Any]:
        """
        Detect concept drift between recent and historical data.

        Args:
            recent_data: Recent data points
            historical_data: Historical baseline data
            significance_level: Statistical significance level

        Returns:
            Dictionary with drift detection results
        """
        result = {
            'drift_detected': False,
            'drift_magnitude': 0.0,
            'drift_type': 'none',
            'confidence': 0.0,
            'statistical_test': 'mean_shift',
            'p_value': 1.0,
            'recommendations': [],
            'error_message': None
        }

        try:
            if not recent_data or not historical_data:
                result['error_message'] = "Empty data provided for drift detection"
                return result

            # Filter valid data
            valid_recent = [float(x) for x in recent_data if isinstance(x, (int, float)) and math.isfinite(x)]
            valid_historical = [float(x) for x in historical_data if isinstance(x, (int, float)) and math.isfinite(x)]

            if len(valid_recent) < 3 or len(valid_historical) < 3:
                result['error_message'] = "Insufficient data for drift detection"
                return result

            # Calculate statistical measures
            recent_mean = statistics.mean(valid_recent)
            historical_mean = statistics.mean(valid_historical)
            recent_std = statistics.stdev(valid_recent) if len(valid_recent) > 1 else 0.0
            historical_std = statistics.stdev(valid_historical) if len(valid_historical) > 1 else 0.0

            # Mean shift detection
            mean_diff = abs(recent_mean - historical_mean)
            pooled_std = math.sqrt((recent_std**2 + historical_std**2) / 2) if (recent_std > 0 or historical_std > 0) else 1.0

            if pooled_std > 0:
                effect_size = mean_diff / pooled_std
                result['drift_magnitude'] = effect_size

                # Simplified statistical test (t-test approximation)
                # For more accuracy, would use scipy.stats.ttest_ind
                if effect_size > 2.0:  # Roughly equivalent to p < 0.05
                    result['drift_detected'] = True
                    result['p_value'] = 0.01  # Approximation
                elif effect_size > 1.0:
                    result['drift_detected'] = True
                    result['p_value'] = 0.1   # Approximation
                else:
                    result['p_value'] = 0.5   # No significant drift

                result['confidence'] = min(1.0, effect_size / 3.0)

                # Determine drift type
                if recent_mean > historical_mean + pooled_std:
                    result['drift_type'] = 'mean_increase'
                elif recent_mean < historical_mean - pooled_std:
                    result['drift_type'] = 'mean_decrease'
                elif recent_std > historical_std * 1.5:
                    result['drift_type'] = 'variance_increase'
                elif recent_std < historical_std * 0.5:
                    result['drift_type'] = 'variance_decrease'

                # Generate recommendations
                if result['drift_detected']:
                    result['recommendations'] = self._generate_drift_recommendations(
                        result['drift_type'], effect_size)

        except Exception as e:
            logger.error(f"Error in concept drift detection: {e}")
            result['error_message'] = f"Drift detection error: {str(e)}"

        return result

    def _select_best_method(self, data: List[float], seasonal_period: Optional[int]) -> ForecastMethod:
        """Select the best forecasting method based on data characteristics."""
        try:
            n = len(data)

            # Check for seasonality
            if seasonal_period and n >= seasonal_period * 2:
                return ForecastMethod.SEASONAL_NAIVE

            # Check for exponential growth
            if all(x > 0 for x in data):
                # Test exponential fit
                log_data = [math.log(x) for x in data]
                linear_r2 = self._calculate_linear_r_squared(list(range(n)), log_data)
                if linear_r2 > 0.8:
                    return ForecastMethod.EXPONENTIAL_SMOOTHING

            # Check for strong linear trend
            linear_r2 = self._calculate_linear_r_squared(list(range(n)), data)
            if linear_r2 > 0.7:
                return ForecastMethod.LINEAR_REGRESSION

            # Default to exponential smoothing for most time series
            return ForecastMethod.EXPONENTIAL_SMOOTHING

        except Exception:
            return ForecastMethod.LINEAR_REGRESSION

    def _calculate_linear_r_squared(self, x_data: List[float], y_data: List[float]) -> float:
        """Calculate R-squared for linear regression."""
        try:
            n = len(x_data)
            if n != len(y_data) or n < 2:
                return 0.0

            x_mean = sum(x_data) / n
            y_mean = sum(y_data) / n

            numerator = sum((x_data[i] - x_mean) * (y_data[i] - y_mean) for i in range(n))
            denominator_x = sum((x_data[i] - x_mean) ** 2 for i in range(n))
            denominator_y = sum((y_data[i] - y_mean) ** 2 for i in range(n))

            if denominator_x == 0 or denominator_y == 0:
                return 0.0

            correlation = numerator / math.sqrt(denominator_x * denominator_y)
            return correlation ** 2

        except Exception:
            return 0.0

    def _forecast_linear_regression(self, data: List[float], horizon: int, result: ForecastResult) -> ForecastResult:
        """Forecast using linear regression."""
        try:
            n = len(data)
            x = list(range(n))

            # Calculate linear regression parameters
            x_mean = sum(x) / n
            y_mean = sum(data) / n

            numerator = sum((x[i] - x_mean) * (data[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

            if abs(denominator) < 1e-10:
                slope = 0.0
            else:
                slope = numerator / denominator

            intercept = y_mean - slope * x_mean

            # Generate predictions
            predictions = []
            for i in range(horizon):
                future_x = n + i
                prediction = intercept + slope * future_x
                predictions.append(prediction)

            result.predictions = predictions
            result.model_parameters = {'slope': slope, 'intercept': intercept}

            # Calculate R-squared
            y_pred = [intercept + slope * xi for xi in x]
            ss_res = sum((data[i] - y_pred[i]) ** 2 for i in range(n))
            ss_tot = sum((data[i] - y_mean) ** 2 for i in range(n))

            if abs(ss_tot) < 1e-10:
                result.r_squared = 1.0 if ss_res < 1e-10 else 0.0
            else:
                result.r_squared = max(0.0, 1.0 - (ss_res / ss_tot))

            # Calculate residuals
            result.residuals = [data[i] - y_pred[i] for i in range(n)]

            # Generate confidence intervals (simplified)
            residual_std = statistics.stdev(result.residuals) if len(result.residuals) > 1 else 0.0
            confidence_margin = 1.96 * residual_std  # 95% confidence interval

            result.confidence_intervals = [
                (pred - confidence_margin, pred + confidence_margin)
                for pred in predictions
            ]

        except Exception as e:
            logger.error(f"Error in linear regression forecasting: {e}")
            result.error_message = f"Linear regression error: {str(e)}"

        return result

    def _forecast_exponential_smoothing(self, data: List[float], horizon: int, result: ForecastResult) -> ForecastResult:
        """Forecast using exponential smoothing."""
        try:
            # Simple exponential smoothing
            alpha = 0.3  # Smoothing parameter

            # Initialize with first observation
            smoothed = [data[0]]

            # Calculate smoothed values
            for i in range(1, len(data)):
                smoothed_value = alpha * data[i] + (1 - alpha) * smoothed[-1]
                smoothed.append(smoothed_value)

            # Generate predictions (use last smoothed value)
            last_smoothed = smoothed[-1]
            predictions = [last_smoothed] * horizon

            result.predictions = predictions
            result.model_parameters = {'alpha': alpha, 'last_smoothed': last_smoothed}

            # Calculate residuals
            result.residuals = [data[i] - smoothed[i] for i in range(len(data))]

            # Calculate MAE
            if result.residuals:
                result.mae = sum(abs(r) for r in result.residuals) / len(result.residuals)

            # Generate confidence intervals
            residual_std = statistics.stdev(result.residuals) if len(result.residuals) > 1 else 0.0
            confidence_margin = 1.96 * residual_std

            result.confidence_intervals = [
                (pred - confidence_margin, pred + confidence_margin)
                for pred in predictions
            ]

        except Exception as e:
            logger.error(f"Error in exponential smoothing forecasting: {e}")
            result.error_message = f"Exponential smoothing error: {str(e)}"

        return result

    def _forecast_moving_average(self, data: List[float], horizon: int, result: ForecastResult) -> ForecastResult:
        """Forecast using moving average."""
        try:
            # Use last 5 points or all data if less than 5
            window_size = min(5, len(data))
            recent_data = data[-window_size:]

            # Calculate moving average
            moving_avg = sum(recent_data) / len(recent_data)

            # Generate predictions (use moving average)
            predictions = [moving_avg] * horizon

            result.predictions = predictions
            result.model_parameters = {'window_size': window_size, 'moving_average': moving_avg}

            # Calculate residuals using moving average
            residuals = []
            for i in range(window_size - 1, len(data)):
                start_idx = max(0, i - window_size + 1)
                window_avg = sum(data[start_idx:i + 1]) / (i - start_idx + 1)
                residuals.append(data[i] - window_avg)

            result.residuals = residuals

            # Calculate MAE
            if result.residuals:
                result.mae = sum(abs(r) for r in result.residuals) / len(result.residuals)

            # Generate confidence intervals
            residual_std = statistics.stdev(result.residuals) if len(result.residuals) > 1 else 0.0
            confidence_margin = 1.96 * residual_std

            result.confidence_intervals = [
                (pred - confidence_margin, pred + confidence_margin)
                for pred in predictions
            ]

        except Exception as e:
            logger.error(f"Error in moving average forecasting: {e}")
            result.error_message = f"Moving average error: {str(e)}"

        return result

    def _forecast_seasonal_naive(self, data: List[float], horizon: int, seasonal_period: Optional[int], result: ForecastResult) -> ForecastResult:
        """Forecast using seasonal naive method."""
        try:
            if not seasonal_period or seasonal_period <= 0:
                seasonal_period = 12  # Default to 12

            # Use last seasonal cycle
            if len(data) >= seasonal_period:
                last_season = data[-seasonal_period:]
            else:
                last_season = data

            # Generate predictions by repeating seasonal pattern
            predictions = []
            for i in range(horizon):
                seasonal_idx = i % len(last_season)
                predictions.append(last_season[seasonal_idx])

            result.predictions = predictions
            result.model_parameters = {'seasonal_period': seasonal_period, 'last_season': last_season}

            # Calculate seasonal components
            result.seasonal_components = last_season

            # Calculate residuals (simplified)
            if len(data) >= seasonal_period * 2:
                residuals = []
                for i in range(seasonal_period, len(data)):
                    seasonal_idx = i % seasonal_period
                    seasonal_value = data[seasonal_idx - seasonal_period] if seasonal_idx < len(data) - seasonal_period else data[seasonal_idx]
                    residuals.append(data[i] - seasonal_value)
                result.residuals = residuals

            # Generate confidence intervals
            if result.residuals:
                residual_std = statistics.stdev(result.residuals)
                confidence_margin = 1.96 * residual_std

                result.confidence_intervals = [
                    (pred - confidence_margin, pred + confidence_margin)
                    for pred in predictions
                ]
            else:
                # Use data standard deviation as fallback
                data_std = statistics.stdev(data) if len(data) > 1 else 0.0
                confidence_margin = 1.96 * data_std

                result.confidence_intervals = [
                    (pred - confidence_margin, pred + confidence_margin)
                    for pred in predictions
                ]

        except Exception as e:
            logger.error(f"Error in seasonal naive forecasting: {e}")
            result.error_message = f"Seasonal naive error: {str(e)}"

        return result

    def _forecast_trend_projection(self, data: List[float], horizon: int, result: ForecastResult) -> ForecastResult:
        """Forecast using trend projection with damping."""
        try:
            n = len(data)

            # Calculate trend using linear regression
            x = list(range(n))
            x_mean = sum(x) / n
            y_mean = sum(data) / n

            numerator = sum((x[i] - x_mean) * (data[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

            if abs(denominator) < 1e-10:
                slope = 0.0
            else:
                slope = numerator / denominator

            intercept = y_mean - slope * x_mean

            # Apply damping to trend for long-term forecasts
            damping_factor = 0.98  # Trend dampens over time

            predictions = []
            for i in range(horizon):
                future_x = n + i
                damped_slope = slope * (damping_factor ** i)
                prediction = intercept + damped_slope * future_x
                predictions.append(prediction)

            result.predictions = predictions
            result.model_parameters = {'slope': slope, 'intercept': intercept, 'damping_factor': damping_factor}

            # Calculate trend component
            result.trend_component = [intercept + slope * xi for xi in x]

            # Calculate residuals
            result.residuals = [data[i] - result.trend_component[i] for i in range(n)]

            # Generate confidence intervals with increasing uncertainty
            residual_std = statistics.stdev(result.residuals) if len(result.residuals) > 1 else 0.0

            confidence_intervals = []
            for i in range(horizon):
                # Increase uncertainty with forecast horizon
                uncertainty_factor = 1 + (i * 0.1)
                confidence_margin = 1.96 * residual_std * uncertainty_factor
                pred = predictions[i]
                confidence_intervals.append((pred - confidence_margin, pred + confidence_margin))

            result.confidence_intervals = confidence_intervals

        except Exception as e:
            logger.error(f"Error in trend projection forecasting: {e}")
            result.error_message = f"Trend projection error: {str(e)}"

        return result

    def _calculate_accuracy_metrics(self, data: List[float], method: ForecastMethod, seasonal_period: Optional[int]) -> Dict[str, float]:
        """Calculate accuracy metrics using cross-validation."""
        try:
            if len(data) < 10:
                return {'mae': 0.0, 'mape': 0.0, 'rmse': 0.0}

            # Use last 20% of data for validation
            split_point = int(len(data) * 0.8)
            train_data = data[:split_point]
            test_data = data[split_point:]

            if len(train_data) < self.min_training_points:
                return {'mae': 0.0, 'mape': 0.0, 'rmse': 0.0}

            # Generate forecast for test period
            dummy_result = ForecastResult(
                predictions=[],
                confidence_intervals=[],
                method_used=method,
                forecast_horizon=len(test_data),
                accuracy_metrics={},
                model_parameters={}
            )

            if method == ForecastMethod.LINEAR_REGRESSION:
                forecast_result = self._forecast_linear_regression(train_data, len(test_data), dummy_result)
            elif method == ForecastMethod.EXPONENTIAL_SMOOTHING:
                forecast_result = self._forecast_exponential_smoothing(train_data, len(test_data), dummy_result)
            elif method == ForecastMethod.MOVING_AVERAGE:
                forecast_result = self._forecast_moving_average(train_data, len(test_data), dummy_result)
            elif method == ForecastMethod.SEASONAL_NAIVE:
                forecast_result = self._forecast_seasonal_naive(train_data, len(test_data), seasonal_period, dummy_result)
            elif method == ForecastMethod.TREND_PROJECTION:
                forecast_result = self._forecast_trend_projection(train_data, len(test_data), dummy_result)
            else:
                return {'mae': 0.0, 'mape': 0.0, 'rmse': 0.0}

            if not forecast_result.predictions or forecast_result.error_message:
                return {'mae': 0.0, 'mape': 0.0, 'rmse': 0.0}

            # Calculate metrics
            predictions = forecast_result.predictions[:len(test_data)]
            errors = [test_data[i] - predictions[i] for i in range(min(len(test_data), len(predictions)))]

            mae = sum(abs(e) for e in errors) / len(errors) if errors else 0.0
            rmse = math.sqrt(sum(e**2 for e in errors) / len(errors)) if errors else 0.0

            # Calculate MAPE (Mean Absolute Percentage Error)
            mape = 0.0
            if errors:
                percentage_errors = []
                for i in range(len(errors)):
                    if abs(test_data[i]) > 1e-10:  # Avoid division by zero
                        pe = abs(errors[i]) / abs(test_data[i]) * 100
                        percentage_errors.append(pe)

                mape = sum(percentage_errors) / len(percentage_errors) if percentage_errors else 0.0

            return {'mae': mae, 'mape': mape, 'rmse': rmse}

        except Exception as e:
            logger.warning(f"Error calculating accuracy metrics: {e}")
            return {'mae': 0.0, 'mape': 0.0, 'rmse': 0.0}

    def _assess_forecast_reliability(self, data: List[float], result: ForecastResult) -> float:
        """Assess the reliability of the forecast."""
        try:
            reliability_score = 0.0

            # Factor 1: Data quality (40% weight)
            data_quality = min(1.0, len(data) / 50.0)  # More data = higher quality
            reliability_score += 0.4 * data_quality

            # Factor 2: Model fit (30% weight)
            if result.r_squared is not None:
                reliability_score += 0.3 * result.r_squared
            elif result.mae is not None and result.mae > 0:
                # Lower MAE = higher reliability (inverse relationship)
                data_range = max(data) - min(data) if len(data) > 1 else 1.0
                normalized_mae = result.mae / data_range if data_range > 0 else 1.0
                fit_score = max(0.0, 1.0 - normalized_mae)
                reliability_score += 0.3 * fit_score

            # Factor 3: Forecast horizon appropriateness (20% weight)
            reasonable_horizon = min(len(data) // 4, 30)  # Max 1/4 of data length or 30 periods
            if result.forecast_horizon <= reasonable_horizon:
                horizon_score = 1.0
            else:
                horizon_score = max(0.0, reasonable_horizon / result.forecast_horizon)
            reliability_score += 0.2 * horizon_score

            # Factor 4: Data stability (10% weight)
            if len(data) > 5:
                recent_std = statistics.stdev(data[-5:])
                overall_std = statistics.stdev(data)
                if overall_std > 0:
                    stability_score = 1.0 - min(1.0, abs(recent_std - overall_std) / overall_std)
                else:
                    stability_score = 1.0
            else:
                stability_score = 0.5  # Neutral score for insufficient data

            reliability_score += 0.1 * stability_score

            return min(1.0, reliability_score)

        except Exception:
            return 0.5  # Default neutral score

    def _generate_forecast_timestamps(self, timestamps: List[datetime], horizon: int) -> List[datetime]:
        """Generate timestamps for forecast periods."""
        try:
            if len(timestamps) < 2:
                # Default to daily intervals
                base_date = timestamps[0] if timestamps else datetime.now()
                return [base_date + timedelta(days=i+1) for i in range(horizon)]

            # Calculate average time interval
            intervals = []
            for i in range(1, len(timestamps)):
                interval = timestamps[i] - timestamps[i-1]
                intervals.append(interval)

            avg_interval = sum(intervals, timedelta()) / len(intervals)

            # Generate future timestamps
            last_timestamp = timestamps[-1]
            future_timestamps = []
            for i in range(horizon):
                future_time = last_timestamp + avg_interval * (i + 1)
                future_timestamps.append(future_time)

            return future_timestamps

        except Exception:
            # Fallback to daily intervals
            base_date = timestamps[-1] if timestamps else datetime.now()
            return [base_date + timedelta(days=i+1) for i in range(horizon)]

    def _generate_scaling_recommendations(self, capacity_prediction: CapacityPrediction, threshold: float) -> List[str]:
        """Generate scaling recommendations based on capacity prediction."""
        recommendations = []

        try:
            max_predicted = max(capacity_prediction.predicted_utilization) if capacity_prediction.predicted_utilization else 0

            if max_predicted >= threshold:
                recommendations.append(f"Scale {capacity_prediction.resource_type} resources immediately - threshold exceeded")

            if capacity_prediction.growth_rate > 0.01:  # 1% growth per period
                recommendations.append(f"Monitor {capacity_prediction.resource_type} growth closely - consistent upward trend detected")

            if capacity_prediction.capacity_exhaustion_date:
                days_to_exhaustion = (capacity_prediction.capacity_exhaustion_date - datetime.now()).days
                if days_to_exhaustion <= 7:
                    recommendations.append(f"Critical: Scale {capacity_prediction.resource_type} within {days_to_exhaustion} days")
                elif days_to_exhaustion <= 30:
                    recommendations.append(f"Plan {capacity_prediction.resource_type} scaling within {days_to_exhaustion} days")

            if capacity_prediction.risk_assessment == "high":
                recommendations.append(f"High risk: Implement auto-scaling for {capacity_prediction.resource_type}")

            if not recommendations:
                recommendations.append(f"{capacity_prediction.resource_type} capacity is adequate for the forecast period")

        except Exception as e:
            logger.warning(f"Error generating scaling recommendations: {e}")
            recommendations.append("Unable to generate specific scaling recommendations")

        return recommendations

    def _predict_bottlenecks(self, historical_data: List[float], predicted_data: List[float], metric_name: str) -> List[str]:
        """Predict potential bottlenecks based on performance trends."""
        bottlenecks = []

        try:
            # Check for degrading trend
            if len(predicted_data) > 0 and len(historical_data) > 0:
                current_avg = statistics.mean(historical_data[-5:]) if len(historical_data) >= 5 else statistics.mean(historical_data)
                future_avg = statistics.mean(predicted_data)

                degradation_ratio = future_avg / current_avg if current_avg > 0 else 1.0

                if metric_name.lower() in ['response_time', 'latency', 'duration'] and degradation_ratio > 1.2:
                    bottlenecks.append(f"Performance degradation predicted - {metric_name} may increase by {((degradation_ratio - 1) * 100):.1f}%")
                elif metric_name.lower() in ['throughput', 'requests_per_second'] and degradation_ratio < 0.8:
                    bottlenecks.append(f"Throughput bottleneck predicted - {metric_name} may decrease by {((1 - degradation_ratio) * 100):.1f}%")

            # Check for high volatility in predictions
            if len(predicted_data) > 1:
                predicted_std = statistics.stdev(predicted_data)
                predicted_mean = statistics.mean(predicted_data)

                if predicted_mean > 0 and (predicted_std / predicted_mean) > 0.3:  # High coefficient of variation
                    bottlenecks.append(f"High volatility predicted in {metric_name} - consider stabilization measures")

        except Exception as e:
            logger.warning(f"Error predicting bottlenecks: {e}")

        return bottlenecks

    def _generate_optimization_recommendations(self, performance_prediction: PerformancePrediction, historical_data: List[float]) -> List[str]:
        """Generate optimization recommendations based on performance prediction."""
        recommendations = []

        try:
            if performance_prediction.performance_trend == "degrading":
                recommendations.append(f"Address performance degradation in {performance_prediction.metric_name}")

                # Specific recommendations based on metric type
                metric_lower = performance_prediction.metric_name.lower()
                if 'memory' in metric_lower:
                    recommendations.append("Consider memory optimization or garbage collection tuning")
                elif 'cpu' in metric_lower:
                    recommendations.append("Consider CPU optimization or load balancing")
                elif 'response_time' in metric_lower or 'latency' in metric_lower:
                    recommendations.append("Consider caching, database optimization, or CDN implementation")
                elif 'throughput' in metric_lower:
                    recommendations.append("Consider horizontal scaling or performance tuning")

            elif performance_prediction.performance_trend == "improving":
                recommendations.append(f"{performance_prediction.metric_name} shows improvement - maintain current optimizations")

            # Check for high variability
            if len(historical_data) > 1:
                cv = statistics.stdev(historical_data) / statistics.mean(historical_data) if statistics.mean(historical_data) > 0 else 0
                if cv > 0.3:  # High coefficient of variation
                    recommendations.append(f"High variability in {performance_prediction.metric_name} - implement monitoring and alerting")

            # Baseline comparison recommendations
            if performance_prediction.baseline_comparison:
                current_vs_baseline = performance_prediction.baseline_comparison.get('current_vs_baseline', 1.0)
                if current_vs_baseline < 0.9:
                    recommendations.append("Performance is below baseline - investigate recent changes")
                elif current_vs_baseline > 1.1:
                    recommendations.append("Performance exceeds baseline - document successful optimizations")

        except Exception as e:
            logger.warning(f"Error generating optimization recommendations: {e}")

        return recommendations

    def _predict_sla_compliance(self, predicted_performance: List[float], baseline_value: float) -> float:
        """Predict SLA compliance based on performance predictions."""
        try:
            if not predicted_performance or baseline_value <= 0:
                return 0.0

            compliant_predictions = sum(1 for pred in predicted_performance if pred >= baseline_value)
            return compliant_predictions / len(predicted_performance)

        except Exception:
            return 0.0

    def _identify_performance_risks(self, historical_data: List[float], predicted_data: List[float]) -> List[str]:
        """Identify performance risks based on data analysis."""
        risks = []

        try:
            # Check for high volatility
            if len(historical_data) > 1:
                cv = statistics.stdev(historical_data) / statistics.mean(historical_data) if statistics.mean(historical_data) > 0 else 0
                if cv > 0.5:
                    risks.append("High performance volatility detected")

            # Check for extreme values in predictions
            if predicted_data:
                mean_pred = statistics.mean(predicted_data)
                std_pred = statistics.stdev(predicted_data) if len(predicted_data) > 1 else 0

                for pred in predicted_data:
                    if std_pred > 0 and abs(pred - mean_pred) > 3 * std_pred:
                        risks.append("Extreme performance values predicted")
                        break

            # Check for consistent degradation
            if len(predicted_data) >= 5:
                recent_trend = predicted_data[-3:]
                early_trend = predicted_data[:3]

                if statistics.mean(recent_trend) < statistics.mean(early_trend) * 0.9:
                    risks.append("Consistent performance degradation trend")

        except Exception as e:
            logger.warning(f"Error identifying performance risks: {e}")

        return risks

    def _generate_drift_recommendations(self, drift_type: str, effect_size: float) -> List[str]:
        """Generate recommendations for handling concept drift."""
        recommendations = []

        try:
            if drift_type == 'mean_increase':
                recommendations.append("Data mean has increased - review for new patterns or external factors")
                if effect_size > 2.0:
                    recommendations.append("Significant mean shift detected - retrain models with recent data")

            elif drift_type == 'mean_decrease':
                recommendations.append("Data mean has decreased - investigate potential system changes")
                if effect_size > 2.0:
                    recommendations.append("Significant mean drop detected - verify data quality and system health")

            elif drift_type == 'variance_increase':
                recommendations.append("Data variability has increased - consider adaptive models")
                recommendations.append("Implement more robust anomaly detection")

            elif drift_type == 'variance_decrease':
                recommendations.append("Data has become more stable - models may be more reliable")
                recommendations.append("Consider tightening prediction intervals")

            # General recommendations
            if effect_size > 1.5:
                recommendations.append("Update model parameters or retrain with recent data")
                recommendations.append("Increase monitoring frequency")

        except Exception as e:
            logger.warning(f"Error generating drift recommendations: {e}")

        return recommendations

    def _is_cached_valid(self, cache_key: str) -> bool:
        """Check if cached result is still valid."""
        if cache_key not in self._model_cache:
            return False

        if cache_key not in self._cache_timestamps:
            return False

        age = datetime.now() - self._cache_timestamps[cache_key]
        return age <= self._cache_ttl

    def _cache_result(self, cache_key: str, result):
        """Cache predictive model result."""
        self._model_cache[cache_key] = result
        self._cache_timestamps[cache_key] = datetime.now()

    def clear_cache(self):
        """Clear all cached results."""
        self._model_cache.clear()
        self._cache_timestamps.clear()