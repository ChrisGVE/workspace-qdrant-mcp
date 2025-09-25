"""
Pattern Recognition Module for trend analysis and pattern detection.

Provides advanced pattern recognition capabilities including:
- Trend detection and analysis
- Seasonality detection
- Cyclic pattern identification
- Change point detection
- Anomaly pattern recognition
- Time series pattern analysis
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


class TrendDirection(Enum):
    """Enumeration for trend directions."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


class PatternType(Enum):
    """Enumeration for pattern types."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    PERIODIC = "periodic"
    SEASONAL = "seasonal"
    RANDOM = "random"
    STEP_CHANGE = "step_change"
    SPIKE = "spike"


@dataclass
class TrendResult:
    """Result container for trend analysis."""

    direction: TrendDirection
    strength: float  # 0.0 to 1.0
    slope: Optional[float] = None
    r_squared: Optional[float] = None
    confidence: float = 0.0  # 0.0 to 1.0
    start_value: Optional[float] = None
    end_value: Optional[float] = None
    change_magnitude: Optional[float] = None
    change_percentage: Optional[float] = None
    duration_periods: int = 0
    is_significant: bool = False
    p_value: Optional[float] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'direction': self.direction.value,
            'strength': self.strength,
            'slope': self.slope,
            'r_squared': self.r_squared,
            'confidence': self.confidence,
            'start_value': self.start_value,
            'end_value': self.end_value,
            'change_magnitude': self.change_magnitude,
            'change_percentage': self.change_percentage,
            'duration_periods': self.duration_periods,
            'is_significant': self.is_significant,
            'p_value': self.p_value,
            'error_message': self.error_message
        }


@dataclass
class PatternResult:
    """Result container for pattern detection."""

    pattern_type: PatternType
    confidence: float  # 0.0 to 1.0
    parameters: Dict[str, Any]
    fit_quality: float  # 0.0 to 1.0
    description: str
    data_points: int
    pattern_strength: float  # 0.0 to 1.0
    start_index: int = 0
    end_index: Optional[int] = None
    is_significant: bool = False
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'pattern_type': self.pattern_type.value,
            'confidence': self.confidence,
            'parameters': self.parameters,
            'fit_quality': self.fit_quality,
            'description': self.description,
            'data_points': self.data_points,
            'pattern_strength': self.pattern_strength,
            'start_index': self.start_index,
            'end_index': self.end_index,
            'is_significant': self.is_significant,
            'error_message': self.error_message
        }


@dataclass
class ChangePointResult:
    """Result container for change point detection."""

    change_points: List[int]
    confidence_scores: List[float]
    change_types: List[str]
    change_magnitudes: List[float]
    statistical_significance: List[bool]
    method_used: str
    total_changes: int
    most_significant_change: Optional[int] = None
    average_confidence: float = 0.0
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'change_points': self.change_points,
            'confidence_scores': self.confidence_scores,
            'change_types': self.change_types,
            'change_magnitudes': self.change_magnitudes,
            'statistical_significance': self.statistical_significance,
            'method_used': self.method_used,
            'total_changes': self.total_changes,
            'most_significant_change': self.most_significant_change,
            'average_confidence': self.average_confidence,
            'error_message': self.error_message
        }


class PatternRecognition:
    """
    Advanced pattern recognition engine for trend and pattern analysis.

    Provides robust pattern detection with comprehensive edge case handling
    for sparse data, noisy signals, and boundary conditions.
    """

    def __init__(self, min_data_points: int = 5, significance_threshold: float = 0.05):
        """
        Initialize pattern recognition engine.

        Args:
            min_data_points: Minimum data points required for analysis
            significance_threshold: Statistical significance threshold (p-value)
        """
        self.min_data_points = min_data_points
        self.significance_threshold = significance_threshold
        self._pattern_cache = {}
        self._cache_ttl = timedelta(minutes=10)
        self._cache_timestamps = {}

    def detect_trend(self, data: List[Union[int, float]],
                    timestamps: Optional[List[datetime]] = None,
                    cache_key: Optional[str] = None) -> TrendResult:
        """
        Detect and analyze trends in time series data.

        Args:
            data: Time series data
            timestamps: Optional timestamps for each data point
            cache_key: Optional cache key for result caching

        Returns:
            TrendResult with comprehensive trend analysis
        """
        # Check cache if key provided
        if cache_key and self._is_cached_valid(cache_key):
            return self._pattern_cache[cache_key]

        result = TrendResult(
            direction=TrendDirection.UNKNOWN,
            strength=0.0,
            duration_periods=len(data) if data else 0
        )

        try:
            # Validate input data
            if not data:
                result.error_message = "Empty dataset provided"
                return result

            if len(data) < self.min_data_points:
                result.error_message = f"Insufficient data points (minimum {self.min_data_points} required)"
                return result

            # Filter and clean data
            valid_data = []
            valid_indices = []

            for i, value in enumerate(data):
                if isinstance(value, (int, float)) and math.isfinite(value):
                    valid_data.append(float(value))
                    valid_indices.append(i)

            if len(valid_data) < self.min_data_points:
                result.error_message = "Insufficient valid data points after filtering"
                return result

            # Calculate linear trend using least squares regression
            n = len(valid_data)
            x_values = list(range(n)) if not timestamps else valid_indices

            # Calculate regression statistics
            x_mean = sum(x_values) / n
            y_mean = sum(valid_data) / n

            numerator = sum((x_values[i] - x_mean) * (valid_data[i] - y_mean) for i in range(n))
            denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))

            if abs(denominator) < 1e-10:  # Avoid division by zero
                result.direction = TrendDirection.STABLE
                result.strength = 0.0
                result.slope = 0.0
            else:
                result.slope = numerator / denominator

                # Calculate R-squared
                y_pred = [y_mean + result.slope * (x_values[i] - x_mean) for i in range(n)]
                ss_res = sum((valid_data[i] - y_pred[i]) ** 2 for i in range(n))
                ss_tot = sum((valid_data[i] - y_mean) ** 2 for i in range(n))

                if abs(ss_tot) < 1e-10:
                    result.r_squared = 1.0 if ss_res < 1e-10 else 0.0
                else:
                    result.r_squared = max(0.0, 1.0 - (ss_res / ss_tot))

            # Determine trend direction and strength
            result.start_value = valid_data[0]
            result.end_value = valid_data[-1]

            if result.slope is not None and abs(result.slope) > 1e-10:
                if result.slope > 0:
                    result.direction = TrendDirection.INCREASING
                else:
                    result.direction = TrendDirection.DECREASING

                # Calculate trend strength based on R-squared and consistency
                result.strength = min(1.0, abs(result.r_squared or 0.0))

                # Calculate change statistics
                result.change_magnitude = result.end_value - result.start_value
                if abs(result.start_value) > 1e-10:
                    result.change_percentage = (result.change_magnitude / result.start_value) * 100
                else:
                    result.change_percentage = float('inf') if result.change_magnitude != 0 else 0.0

            else:
                result.direction = TrendDirection.STABLE
                result.strength = 0.0
                result.change_magnitude = 0.0
                result.change_percentage = 0.0

            # Assess volatility
            if result.r_squared is not None and result.r_squared < 0.3 and self._calculate_volatility(valid_data) > 0.5:
                result.direction = TrendDirection.VOLATILE

            # Calculate confidence and significance
            result.confidence = self._calculate_trend_confidence(valid_data, result)
            result.is_significant = result.confidence > (1.0 - self.significance_threshold)

            # Calculate p-value for trend significance
            if result.slope is not None and result.r_squared is not None and n > 2:
                result.p_value = self._calculate_trend_p_value(valid_data, result.slope, n)

            # Cache result if key provided
            if cache_key:
                self._cache_result(cache_key, result)

        except Exception as e:
            logger.error(f"Error in trend detection: {e}")
            result.error_message = f"Trend detection error: {str(e)}"

        return result

    def detect_patterns(self, data: List[Union[int, float]],
                       pattern_types: Optional[List[PatternType]] = None,
                       min_pattern_length: int = 3) -> List[PatternResult]:
        """
        Detect various patterns in the data.

        Args:
            data: Input data series
            pattern_types: Specific pattern types to look for (None = all)
            min_pattern_length: Minimum length for pattern detection

        Returns:
            List of detected patterns with confidence scores
        """
        results = []

        try:
            if not data or len(data) < min_pattern_length:
                logger.warning(f"Insufficient data for pattern detection (minimum {min_pattern_length} points)")
                return results

            # Filter valid data
            valid_data = []
            for value in data:
                if isinstance(value, (int, float)) and math.isfinite(value):
                    valid_data.append(float(value))

            if len(valid_data) < min_pattern_length:
                logger.warning("Insufficient valid data points for pattern detection")
                return results

            # Default to all pattern types if none specified
            if pattern_types is None:
                pattern_types = list(PatternType)

            # Detect each pattern type
            for pattern_type in pattern_types:
                try:
                    pattern_result = self._detect_specific_pattern(valid_data, pattern_type)
                    if pattern_result and pattern_result.confidence > 0.1:  # Only include meaningful patterns
                        results.append(pattern_result)
                except Exception as e:
                    logger.warning(f"Error detecting {pattern_type.value} pattern: {e}")
                    continue

            # Sort by confidence (highest first)
            results.sort(key=lambda x: x.confidence, reverse=True)

        except Exception as e:
            logger.error(f"Error in pattern detection: {e}")

        return results

    def detect_change_points(self, data: List[Union[int, float]],
                           method: str = "variance",
                           min_segment_length: int = 5) -> ChangePointResult:
        """
        Detect change points in time series data.

        Args:
            data: Input time series data
            method: Change point detection method ('variance', 'mean', 'trend')
            min_segment_length: Minimum length between change points

        Returns:
            ChangePointResult with detected change points
        """
        result = ChangePointResult(
            change_points=[],
            confidence_scores=[],
            change_types=[],
            change_magnitudes=[],
            statistical_significance=[],
            method_used=method,
            total_changes=0
        )

        try:
            if not data or len(data) < min_segment_length * 2:
                result.error_message = f"Insufficient data for change point detection (minimum {min_segment_length * 2} points)"
                return result

            # Filter valid data
            valid_data = []
            valid_indices = []

            for i, value in enumerate(data):
                if isinstance(value, (int, float)) and math.isfinite(value):
                    valid_data.append(float(value))
                    valid_indices.append(i)

            if len(valid_data) < min_segment_length * 2:
                result.error_message = "Insufficient valid data points after filtering"
                return result

            # Apply change point detection method
            if method == "variance":
                change_points = self._detect_variance_change_points(valid_data, min_segment_length)
            elif method == "mean":
                change_points = self._detect_mean_change_points(valid_data, min_segment_length)
            elif method == "trend":
                change_points = self._detect_trend_change_points(valid_data, min_segment_length)
            else:
                result.error_message = f"Unknown change point detection method: {method}"
                return result

            # Process detected change points
            result.change_points = [valid_indices[cp] for cp in change_points if cp < len(valid_indices)]
            result.total_changes = len(result.change_points)

            # Calculate confidence scores and change characteristics
            for cp in change_points:
                if cp < len(valid_data) - min_segment_length and cp >= min_segment_length:
                    confidence = self._calculate_change_point_confidence(valid_data, cp, method)
                    change_type = self._classify_change_type(valid_data, cp)
                    magnitude = self._calculate_change_magnitude(valid_data, cp)

                    result.confidence_scores.append(confidence)
                    result.change_types.append(change_type)
                    result.change_magnitudes.append(magnitude)
                    result.statistical_significance.append(confidence > (1.0 - self.significance_threshold))

            # Find most significant change
            if result.confidence_scores:
                max_confidence_idx = result.confidence_scores.index(max(result.confidence_scores))
                result.most_significant_change = result.change_points[max_confidence_idx]
                result.average_confidence = sum(result.confidence_scores) / len(result.confidence_scores)

        except Exception as e:
            logger.error(f"Error in change point detection: {e}")
            result.error_message = f"Change point detection error: {str(e)}"

        return result

    def detect_seasonality(self, data: List[Union[int, float]],
                          period_hints: Optional[List[int]] = None,
                          max_period: Optional[int] = None) -> Dict[str, Any]:
        """
        Detect seasonal patterns in time series data.

        Args:
            data: Input time series data
            period_hints: Suggested periods to check for seasonality
            max_period: Maximum period to search for

        Returns:
            Dictionary with seasonality analysis results
        """
        result = {
            'seasonal_periods': [],
            'seasonal_strengths': [],
            'dominant_period': None,
            'dominant_strength': 0.0,
            'is_seasonal': False,
            'confidence': 0.0,
            'method': 'autocorrelation',
            'error_message': None
        }

        try:
            if not data or len(data) < 10:
                result['error_message'] = "Insufficient data for seasonality detection"
                return result

            # Filter valid data
            valid_data = []
            for value in data:
                if isinstance(value, (int, float)) and math.isfinite(value):
                    valid_data.append(float(value))

            if len(valid_data) < 10:
                result['error_message'] = "Insufficient valid data points for seasonality detection"
                return result

            # Set default max period
            if max_period is None:
                max_period = min(len(valid_data) // 3, 50)

            # Set default period hints if not provided
            if period_hints is None:
                period_hints = list(range(2, min(max_period + 1, len(valid_data) // 2)))

            # Calculate autocorrelation for different periods
            seasonal_scores = {}

            for period in period_hints:
                if period >= len(valid_data) // 2:
                    continue

                try:
                    autocorr = self._calculate_autocorrelation(valid_data, period)
                    if autocorr is not None:
                        seasonal_scores[period] = abs(autocorr)
                except Exception as e:
                    logger.warning(f"Error calculating autocorrelation for period {period}: {e}")
                    continue

            if not seasonal_scores:
                result['error_message'] = "Could not calculate seasonal scores"
                return result

            # Find significant seasonal periods
            threshold = 0.3  # Minimum correlation for seasonality
            for period, strength in seasonal_scores.items():
                if strength > threshold:
                    result['seasonal_periods'].append(period)
                    result['seasonal_strengths'].append(strength)

            # Determine dominant period
            if seasonal_scores:
                dominant_period = max(seasonal_scores.keys(), key=lambda k: seasonal_scores[k])
                result['dominant_period'] = dominant_period
                result['dominant_strength'] = seasonal_scores[dominant_period]
                result['is_seasonal'] = result['dominant_strength'] > threshold
                result['confidence'] = min(1.0, result['dominant_strength'] * 2)  # Scale confidence

        except Exception as e:
            logger.error(f"Error in seasonality detection: {e}")
            result['error_message'] = f"Seasonality detection error: {str(e)}"

        return result

    def _detect_specific_pattern(self, data: List[float], pattern_type: PatternType) -> Optional[PatternResult]:
        """Detect a specific pattern type in the data."""
        try:
            if pattern_type == PatternType.LINEAR:
                return self._detect_linear_pattern(data)
            elif pattern_type == PatternType.EXPONENTIAL:
                return self._detect_exponential_pattern(data)
            elif pattern_type == PatternType.LOGARITHMIC:
                return self._detect_logarithmic_pattern(data)
            elif pattern_type == PatternType.PERIODIC:
                return self._detect_periodic_pattern(data)
            elif pattern_type == PatternType.STEP_CHANGE:
                return self._detect_step_change_pattern(data)
            elif pattern_type == PatternType.SPIKE:
                return self._detect_spike_pattern(data)
            else:
                return None

        except Exception as e:
            logger.warning(f"Error detecting {pattern_type.value} pattern: {e}")
            return None

    def _detect_linear_pattern(self, data: List[float]) -> Optional[PatternResult]:
        """Detect linear patterns in the data."""
        try:
            n = len(data)
            if n < 3:
                return None

            x = list(range(n))

            # Calculate linear regression
            x_mean = sum(x) / n
            y_mean = sum(data) / n

            numerator = sum((x[i] - x_mean) * (data[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

            if abs(denominator) < 1e-10:
                return None

            slope = numerator / denominator
            intercept = y_mean - slope * x_mean

            # Calculate R-squared
            y_pred = [intercept + slope * xi for xi in x]
            ss_res = sum((data[i] - y_pred[i]) ** 2 for i in range(n))
            ss_tot = sum((data[i] - y_mean) ** 2 for i in range(n))

            if abs(ss_tot) < 1e-10:
                r_squared = 1.0 if ss_res < 1e-10 else 0.0
            else:
                r_squared = max(0.0, 1.0 - (ss_res / ss_tot))

            return PatternResult(
                pattern_type=PatternType.LINEAR,
                confidence=r_squared,
                parameters={'slope': slope, 'intercept': intercept},
                fit_quality=r_squared,
                description=f"Linear trend with slope {slope:.4f}",
                data_points=n,
                pattern_strength=r_squared,
                is_significant=r_squared > 0.7
            )

        except Exception as e:
            logger.warning(f"Error in linear pattern detection: {e}")
            return None

    def _detect_exponential_pattern(self, data: List[float]) -> Optional[PatternResult]:
        """Detect exponential patterns in the data."""
        try:
            n = len(data)
            if n < 3:
                return None

            # Check if all values are positive (required for log transformation)
            if any(val <= 0 for val in data):
                return None

            # Transform to log space for exponential pattern detection
            log_data = [math.log(val) for val in data]
            x = list(range(n))

            # Linear regression in log space
            x_mean = sum(x) / n
            y_mean = sum(log_data) / n

            numerator = sum((x[i] - x_mean) * (log_data[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

            if abs(denominator) < 1e-10:
                return None

            slope = numerator / denominator
            intercept = y_mean - slope * x_mean

            # Calculate R-squared in log space
            y_pred = [intercept + slope * xi for xi in x]
            ss_res = sum((log_data[i] - y_pred[i]) ** 2 for i in range(n))
            ss_tot = sum((log_data[i] - y_mean) ** 2 for i in range(n))

            if abs(ss_tot) < 1e-10:
                r_squared = 1.0 if ss_res < 1e-10 else 0.0
            else:
                r_squared = max(0.0, 1.0 - (ss_res / ss_tot))

            # Transform back to original space
            base = math.exp(intercept)
            growth_rate = slope

            return PatternResult(
                pattern_type=PatternType.EXPONENTIAL,
                confidence=r_squared,
                parameters={'base': base, 'growth_rate': growth_rate},
                fit_quality=r_squared,
                description=f"Exponential pattern with growth rate {growth_rate:.4f}",
                data_points=n,
                pattern_strength=r_squared,
                is_significant=r_squared > 0.7
            )

        except Exception as e:
            logger.warning(f"Error in exponential pattern detection: {e}")
            return None

    def _detect_logarithmic_pattern(self, data: List[float]) -> Optional[PatternResult]:
        """Detect logarithmic patterns in the data."""
        try:
            n = len(data)
            if n < 3:
                return None

            # Create log-transformed x values (x must be positive)
            x = [math.log(i + 1) for i in range(n)]

            # Linear regression with log-transformed x
            x_mean = sum(x) / n
            y_mean = sum(data) / n

            numerator = sum((x[i] - x_mean) * (data[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

            if abs(denominator) < 1e-10:
                return None

            slope = numerator / denominator
            intercept = y_mean - slope * x_mean

            # Calculate R-squared
            y_pred = [intercept + slope * xi for xi in x]
            ss_res = sum((data[i] - y_pred[i]) ** 2 for i in range(n))
            ss_tot = sum((data[i] - y_mean) ** 2 for i in range(n))

            if abs(ss_tot) < 1e-10:
                r_squared = 1.0 if ss_res < 1e-10 else 0.0
            else:
                r_squared = max(0.0, 1.0 - (ss_res / ss_tot))

            return PatternResult(
                pattern_type=PatternType.LOGARITHMIC,
                confidence=r_squared,
                parameters={'slope': slope, 'intercept': intercept},
                fit_quality=r_squared,
                description=f"Logarithmic pattern with slope {slope:.4f}",
                data_points=n,
                pattern_strength=r_squared,
                is_significant=r_squared > 0.7
            )

        except Exception as e:
            logger.warning(f"Error in logarithmic pattern detection: {e}")
            return None

    def _detect_periodic_pattern(self, data: List[float]) -> Optional[PatternResult]:
        """Detect periodic patterns using autocorrelation."""
        try:
            n = len(data)
            if n < 6:
                return None

            best_period = None
            best_correlation = 0.0

            # Check periods from 2 to n//3
            max_period = min(n // 3, 20)

            for period in range(2, max_period + 1):
                correlation = self._calculate_autocorrelation(data, period)
                if correlation is not None and abs(correlation) > abs(best_correlation):
                    best_correlation = correlation
                    best_period = period

            if best_period and abs(best_correlation) > 0.3:
                return PatternResult(
                    pattern_type=PatternType.PERIODIC,
                    confidence=abs(best_correlation),
                    parameters={'period': best_period, 'correlation': best_correlation},
                    fit_quality=abs(best_correlation),
                    description=f"Periodic pattern with period {best_period}",
                    data_points=n,
                    pattern_strength=abs(best_correlation),
                    is_significant=abs(best_correlation) > 0.5
                )

            return None

        except Exception as e:
            logger.warning(f"Error in periodic pattern detection: {e}")
            return None

    def _detect_step_change_pattern(self, data: List[float]) -> Optional[PatternResult]:
        """Detect step change patterns."""
        try:
            n = len(data)
            if n < 6:
                return None

            max_change = 0.0
            best_change_point = None

            # Look for the largest step change
            for i in range(2, n - 2):
                before_mean = sum(data[:i]) / i
                after_mean = sum(data[i:]) / (n - i)
                change = abs(after_mean - before_mean)

                if change > max_change:
                    max_change = change
                    best_change_point = i

            if best_change_point is not None:
                # Calculate confidence based on the magnitude of change relative to noise
                data_std = statistics.stdev(data) if len(data) > 1 else 1.0
                confidence = min(1.0, max_change / (data_std + 1e-10))

                return PatternResult(
                    pattern_type=PatternType.STEP_CHANGE,
                    confidence=confidence,
                    parameters={'change_point': best_change_point, 'magnitude': max_change},
                    fit_quality=confidence,
                    description=f"Step change at position {best_change_point}",
                    data_points=n,
                    pattern_strength=confidence,
                    start_index=best_change_point,
                    is_significant=confidence > 0.5
                )

            return None

        except Exception as e:
            logger.warning(f"Error in step change pattern detection: {e}")
            return None

    def _detect_spike_pattern(self, data: List[float]) -> Optional[PatternResult]:
        """Detect spike patterns (outliers)."""
        try:
            n = len(data)
            if n < 5:
                return None

            mean_val = statistics.mean(data)
            std_val = statistics.stdev(data) if n > 1 else 0.0

            if std_val == 0:
                return None

            # Find spikes (values more than 2 std deviations from mean)
            spikes = []
            spike_magnitudes = []

            for i, value in enumerate(data):
                z_score = abs((value - mean_val) / std_val)
                if z_score > 2.0:
                    spikes.append(i)
                    spike_magnitudes.append(z_score)

            if spikes:
                max_spike_magnitude = max(spike_magnitudes)
                confidence = min(1.0, (max_spike_magnitude - 2.0) / 3.0)  # Scale from 2-5 sigma to 0-1

                return PatternResult(
                    pattern_type=PatternType.SPIKE,
                    confidence=confidence,
                    parameters={'spike_positions': spikes, 'magnitudes': spike_magnitudes},
                    fit_quality=confidence,
                    description=f"Spike pattern with {len(spikes)} spikes",
                    data_points=n,
                    pattern_strength=confidence,
                    is_significant=max_spike_magnitude > 3.0
                )

            return None

        except Exception as e:
            logger.warning(f"Error in spike pattern detection: {e}")
            return None

    def _detect_variance_change_points(self, data: List[float], min_segment_length: int) -> List[int]:
        """Detect change points based on variance changes."""
        change_points = []
        n = len(data)

        for i in range(min_segment_length, n - min_segment_length):
            left_segment = data[:i]
            right_segment = data[i:]

            if len(left_segment) > 1 and len(right_segment) > 1:
                left_var = statistics.variance(left_segment)
                right_var = statistics.variance(right_segment)

                # Check for significant variance change
                variance_ratio = max(left_var, right_var) / (min(left_var, right_var) + 1e-10)
                if variance_ratio > 2.0:  # Arbitrary threshold
                    change_points.append(i)

        return change_points

    def _detect_mean_change_points(self, data: List[float], min_segment_length: int) -> List[int]:
        """Detect change points based on mean changes."""
        change_points = []
        n = len(data)

        for i in range(min_segment_length, n - min_segment_length):
            left_segment = data[:i]
            right_segment = data[i:]

            left_mean = statistics.mean(left_segment)
            right_mean = statistics.mean(right_segment)

            # Calculate pooled standard deviation
            left_var = statistics.variance(left_segment) if len(left_segment) > 1 else 0
            right_var = statistics.variance(right_segment) if len(right_segment) > 1 else 0

            pooled_std = math.sqrt(((len(left_segment) - 1) * left_var +
                                  (len(right_segment) - 1) * right_var) /
                                 (len(left_segment) + len(right_segment) - 2))

            if pooled_std > 0:
                # Check for significant mean change (t-test like)
                mean_diff = abs(left_mean - right_mean)
                threshold = 2.0 * pooled_std  # Arbitrary threshold

                if mean_diff > threshold:
                    change_points.append(i)

        return change_points

    def _detect_trend_change_points(self, data: List[float], min_segment_length: int) -> List[int]:
        """Detect change points based on trend changes."""
        change_points = []
        n = len(data)

        for i in range(min_segment_length, n - min_segment_length):
            if i + min_segment_length < n:
                left_segment = data[max(0, i - min_segment_length):i]
                right_segment = data[i:min(n, i + min_segment_length)]

                # Calculate trends for both segments
                left_trend = self._calculate_segment_trend(left_segment)
                right_trend = self._calculate_segment_trend(right_segment)

                # Check for trend direction change
                if left_trend is not None and right_trend is not None:
                    if (left_trend > 0 and right_trend < 0) or (left_trend < 0 and right_trend > 0):
                        change_points.append(i)
                    elif abs(left_trend - right_trend) > abs(left_trend) * 0.5:  # Significant trend change
                        change_points.append(i)

        return change_points

    def _calculate_segment_trend(self, segment: List[float]) -> Optional[float]:
        """Calculate trend (slope) for a data segment."""
        try:
            n = len(segment)
            if n < 2:
                return None

            x = list(range(n))
            x_mean = sum(x) / n
            y_mean = sum(segment) / n

            numerator = sum((x[i] - x_mean) * (segment[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

            if abs(denominator) < 1e-10:
                return 0.0

            return numerator / denominator

        except Exception:
            return None

    def _calculate_autocorrelation(self, data: List[float], lag: int) -> Optional[float]:
        """Calculate autocorrelation at specified lag."""
        try:
            n = len(data)
            if lag >= n or lag < 1:
                return None

            mean_val = statistics.mean(data)

            # Calculate numerator and denominator for autocorrelation
            numerator = sum((data[i] - mean_val) * (data[i + lag] - mean_val)
                           for i in range(n - lag))

            denominator = sum((data[i] - mean_val) ** 2 for i in range(n))

            if abs(denominator) < 1e-10:
                return None

            return numerator / denominator

        except Exception as e:
            logger.warning(f"Error calculating autocorrelation: {e}")
            return None

    def _calculate_volatility(self, data: List[float]) -> float:
        """Calculate data volatility (coefficient of variation)."""
        try:
            if len(data) < 2:
                return 0.0

            mean_val = statistics.mean(data)
            std_val = statistics.stdev(data)

            if abs(mean_val) < 1e-10:
                return float('inf') if std_val > 0 else 0.0

            return std_val / abs(mean_val)

        except Exception:
            return 0.0

    def _calculate_trend_confidence(self, data: List[float], trend_result: TrendResult) -> float:
        """Calculate confidence in trend analysis."""
        try:
            if trend_result.r_squared is None:
                return 0.0

            # Base confidence on R-squared
            confidence = trend_result.r_squared

            # Adjust for data length (more data = more confidence)
            length_factor = min(1.0, len(data) / 20.0)
            confidence *= length_factor

            # Adjust for trend consistency
            if trend_result.slope is not None:
                consistency = self._calculate_trend_consistency(data)
                confidence *= consistency

            return min(1.0, confidence)

        except Exception:
            return 0.0

    def _calculate_trend_consistency(self, data: List[float]) -> float:
        """Calculate how consistent the trend is throughout the data."""
        try:
            n = len(data)
            if n < 4:
                return 1.0

            # Divide data into segments and check trend direction consistency
            segment_size = max(3, n // 4)
            trends = []

            for i in range(0, n - segment_size + 1, segment_size // 2):
                segment = data[i:i + segment_size]
                if len(segment) >= 3:
                    trend = self._calculate_segment_trend(segment)
                    if trend is not None:
                        trends.append(1 if trend > 0 else -1 if trend < 0 else 0)

            if not trends:
                return 0.0

            # Calculate consistency as agreement in trend direction
            most_common = max(set(trends), key=trends.count)
            consistency = trends.count(most_common) / len(trends)

            return consistency

        except Exception:
            return 0.0

    def _calculate_trend_p_value(self, data: List[float], slope: float, n: int) -> float:
        """Calculate p-value for trend significance."""
        try:
            # Simplified p-value calculation based on t-test
            # This is an approximation - for more accurate results, use scipy.stats

            if n <= 2:
                return 1.0

            # Calculate standard error of slope
            x = list(range(n))
            y_mean = statistics.mean(data)
            x_mean = statistics.mean(x)

            # Residual sum of squares
            y_pred = [y_mean + slope * (xi - x_mean) for xi in x]
            ss_res = sum((data[i] - y_pred[i]) ** 2 for i in range(n))

            # Degrees of freedom
            df = n - 2

            if df <= 0 or ss_res <= 0:
                return 1.0

            # Standard error
            mse = ss_res / df
            ss_x = sum((xi - x_mean) ** 2 for xi in x)

            if ss_x <= 0:
                return 1.0

            se_slope = math.sqrt(mse / ss_x)

            # t-statistic
            if se_slope <= 0:
                return 1.0

            t_stat = abs(slope) / se_slope

            # Rough p-value approximation (this is very simplified)
            # For more accuracy, would need proper t-distribution CDF
            if t_stat > 3.0:
                return 0.01
            elif t_stat > 2.0:
                return 0.05
            elif t_stat > 1.0:
                return 0.20
            else:
                return 0.50

        except Exception:
            return 1.0

    def _calculate_change_point_confidence(self, data: List[float], change_point: int, method: str) -> float:
        """Calculate confidence for a detected change point."""
        try:
            n = len(data)
            if change_point <= 0 or change_point >= n:
                return 0.0

            before_segment = data[:change_point]
            after_segment = data[change_point:]

            if len(before_segment) < 2 or len(after_segment) < 2:
                return 0.0

            if method == "variance":
                before_var = statistics.variance(before_segment)
                after_var = statistics.variance(after_segment)
                ratio = max(before_var, after_var) / (min(before_var, after_var) + 1e-10)
                return min(1.0, (ratio - 1.0) / 4.0)  # Scale to 0-1

            elif method == "mean":
                before_mean = statistics.mean(before_segment)
                after_mean = statistics.mean(after_segment)

                # Pooled standard deviation
                before_var = statistics.variance(before_segment)
                after_var = statistics.variance(after_segment)
                pooled_std = math.sqrt(((len(before_segment) - 1) * before_var +
                                      (len(after_segment) - 1) * after_var) /
                                     (len(before_segment) + len(after_segment) - 2))

                if pooled_std <= 0:
                    return 0.0

                effect_size = abs(before_mean - after_mean) / pooled_std
                return min(1.0, effect_size / 3.0)  # Scale to 0-1

            elif method == "trend":
                before_trend = self._calculate_segment_trend(before_segment)
                after_trend = self._calculate_segment_trend(after_segment)

                if before_trend is None or after_trend is None:
                    return 0.0

                trend_change = abs(before_trend - after_trend)
                return min(1.0, trend_change / 2.0)  # Scale to 0-1

            return 0.0

        except Exception:
            return 0.0

    def _classify_change_type(self, data: List[float], change_point: int) -> str:
        """Classify the type of change at a change point."""
        try:
            before_segment = data[:change_point]
            after_segment = data[change_point:]

            if len(before_segment) < 2 or len(after_segment) < 2:
                return "unknown"

            before_mean = statistics.mean(before_segment)
            after_mean = statistics.mean(after_segment)

            before_var = statistics.variance(before_segment)
            after_var = statistics.variance(after_segment)

            # Classify based on changes in mean and variance
            mean_change = abs(after_mean - before_mean)
            var_ratio = max(before_var, after_var) / (min(before_var, after_var) + 1e-10)

            if mean_change > abs(before_mean) * 0.1:  # Significant mean change
                if after_mean > before_mean:
                    return "level_increase"
                else:
                    return "level_decrease"
            elif var_ratio > 2.0:  # Significant variance change
                if after_var > before_var:
                    return "variance_increase"
                else:
                    return "variance_decrease"
            else:
                return "trend_change"

        except Exception:
            return "unknown"

    def _calculate_change_magnitude(self, data: List[float], change_point: int) -> float:
        """Calculate the magnitude of change at a change point."""
        try:
            before_segment = data[:change_point]
            after_segment = data[change_point:]

            if len(before_segment) < 1 or len(after_segment) < 1:
                return 0.0

            before_mean = statistics.mean(before_segment)
            after_mean = statistics.mean(after_segment)

            return abs(after_mean - before_mean)

        except Exception:
            return 0.0

    def _is_cached_valid(self, cache_key: str) -> bool:
        """Check if cached result is still valid."""
        if cache_key not in self._pattern_cache:
            return False

        if cache_key not in self._cache_timestamps:
            return False

        age = datetime.now() - self._cache_timestamps[cache_key]
        return age <= self._cache_ttl

    def _cache_result(self, cache_key: str, result):
        """Cache pattern recognition result."""
        self._pattern_cache[cache_key] = result
        self._cache_timestamps[cache_key] = datetime.now()

    def clear_cache(self):
        """Clear all cached results."""
        self._pattern_cache.clear()
        self._cache_timestamps.clear()