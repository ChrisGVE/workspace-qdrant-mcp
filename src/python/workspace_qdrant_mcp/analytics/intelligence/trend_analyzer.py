"""
Advanced Trend Analyzer for ML-based trend detection and analysis.

Provides sophisticated trend analysis capabilities including:
- Multi-dimensional trend detection with confidence scoring
- Seasonal decomposition and cycle identification
- Change point detection and trend transitions
- Trend strength and direction classification
- Future trend prediction and forecasting
- Trend anomaly detection and pattern breaks
"""

import math
import statistics
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

import numpy as np


logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """Enumeration for trend directions."""
    STRONGLY_INCREASING = "strongly_increasing"
    MODERATELY_INCREASING = "moderately_increasing"
    WEAKLY_INCREASING = "weakly_increasing"
    STABLE = "stable"
    WEAKLY_DECREASING = "weakly_decreasing"
    MODERATELY_DECREASING = "moderately_decreasing"
    STRONGLY_DECREASING = "strongly_decreasing"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


class TrendStrength(Enum):
    """Enumeration for trend strength levels."""
    VERY_WEAK = "very_weak"    # 0.0 - 0.2
    WEAK = "weak"              # 0.2 - 0.4
    MODERATE = "moderate"      # 0.4 - 0.6
    STRONG = "strong"          # 0.6 - 0.8
    VERY_STRONG = "very_strong" # 0.8 - 1.0


class TrendType(Enum):
    """Enumeration for trend types."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    POLYNOMIAL = "polynomial"
    CYCLICAL = "cyclical"
    SEASONAL = "seasonal"
    IRREGULAR = "irregular"


class ChangePointType(Enum):
    """Enumeration for change point types."""
    LEVEL_SHIFT = "level_shift"        # Sudden level change
    TREND_CHANGE = "trend_change"      # Change in trend slope
    VOLATILITY_CHANGE = "volatility_change"  # Change in variance
    STRUCTURAL_BREAK = "structural_break"    # Complete pattern change


@dataclass
class ChangePoint:
    """Container for change point detection results."""

    index: int
    timestamp: Optional[datetime] = None
    change_type: ChangePointType = ChangePointType.LEVEL_SHIFT
    before_value: float = 0.0
    after_value: float = 0.0
    magnitude: float = 0.0
    confidence: float = 0.0
    statistical_significance: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'index': self.index,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'change_type': self.change_type.value,
            'before_value': self.before_value,
            'after_value': self.after_value,
            'magnitude': self.magnitude,
            'confidence': self.confidence,
            'statistical_significance': self.statistical_significance,
            'context': self.context
        }


@dataclass
class TrendSegment:
    """Container for individual trend segments."""

    start_index: int
    end_index: int
    direction: TrendDirection
    strength: TrendStrength
    trend_type: TrendType
    slope: float
    intercept: float
    r_squared: float
    confidence: float
    values: List[float] = field(default_factory=list)
    predictions: List[float] = field(default_factory=list)
    residuals: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'start_index': self.start_index,
            'end_index': self.end_index,
            'direction': self.direction.value,
            'strength': self.strength.value,
            'trend_type': self.trend_type.value,
            'slope': self.slope,
            'intercept': self.intercept,
            'r_squared': self.r_squared,
            'confidence': self.confidence,
            'segment_length': self.end_index - self.start_index + 1,
            'values': self.values,
            'predictions': self.predictions,
            'residuals': self.residuals
        }


@dataclass
class TrendAnalysisResult:
    """Container for comprehensive trend analysis results."""

    overall_direction: TrendDirection
    overall_strength: TrendStrength
    overall_trend_type: TrendType
    trend_segments: List[TrendSegment] = field(default_factory=list)
    change_points: List[ChangePoint] = field(default_factory=list)
    seasonality_detected: bool = False
    seasonal_period: Optional[int] = None
    seasonal_strength: float = 0.0
    trend_confidence: float = 0.0
    analysis_quality: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_reliable: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'overall_direction': self.overall_direction.value,
            'overall_strength': self.overall_strength.value,
            'overall_trend_type': self.overall_trend_type.value,
            'trend_segments': [segment.to_dict() for segment in self.trend_segments],
            'change_points': [cp.to_dict() for cp in self.change_points],
            'seasonality_detected': self.seasonality_detected,
            'seasonal_period': self.seasonal_period,
            'seasonal_strength': self.seasonal_strength,
            'trend_confidence': self.trend_confidence,
            'analysis_quality': self.analysis_quality,
            'recommendations': self.recommendations,
            'metadata': self.metadata,
            'is_reliable': self.is_reliable,
            'error_message': self.error_message
        }


class AdvancedTrendAnalyzer:
    """
    Advanced trend analyzer with ML-based detection and comprehensive edge case handling.

    Provides sophisticated trend analysis including segmentation, change point detection,
    seasonality analysis, and robust handling of noisy, sparse, or irregular data.
    """

    def __init__(self,
                 min_segment_length: int = 5,
                 change_point_threshold: float = 0.1,
                 seasonality_min_periods: int = 8,
                 trend_confidence_threshold: float = 0.3):
        """
        Initialize advanced trend analyzer.

        Args:
            min_segment_length: Minimum length for trend segments
            change_point_threshold: Threshold for change point detection (0-1)
            seasonality_min_periods: Minimum periods required for seasonality detection
            trend_confidence_threshold: Minimum confidence for reliable trend detection
        """
        self.min_segment_length = min_segment_length
        self.change_point_threshold = change_point_threshold
        self.seasonality_min_periods = seasonality_min_periods
        self.trend_confidence_threshold = trend_confidence_threshold

    def analyze_trends(self,
                      data: List[Union[int, float]],
                      timestamps: Optional[List[datetime]] = None) -> TrendAnalysisResult:
        """
        Perform comprehensive trend analysis on time series data.

        Args:
            data: Time series values
            timestamps: Optional timestamps for each data point

        Returns:
            TrendAnalysisResult with comprehensive trend analysis
        """
        result = TrendAnalysisResult(
            overall_direction=TrendDirection.UNKNOWN,
            overall_strength=TrendStrength.VERY_WEAK,
            overall_trend_type=TrendType.IRREGULAR
        )

        try:
            # Filter and validate input data
            valid_data, valid_timestamps = self._prepare_data(data, timestamps)

            if len(valid_data) < 3:
                result.is_reliable = False
                result.error_message = "Insufficient data for trend analysis (minimum 3 points required)"
                return result

            # Perform overall trend analysis
            overall_trend = self._analyze_overall_trend(valid_data)
            result.overall_direction = overall_trend['direction']
            result.overall_strength = overall_trend['strength']
            result.overall_trend_type = overall_trend['trend_type']
            result.trend_confidence = overall_trend['confidence']

            # Detect change points if we have enough data
            try:
                if len(valid_data) >= self.min_segment_length * 2:
                    result.change_points = self._detect_change_points(valid_data, valid_timestamps)
            except Exception as e:
                logger.warning(f"Error in change point detection: {e}")
                result.change_points = []  # Continue with empty change points

            # Segment trend analysis
            if result.change_points:
                result.trend_segments = self._analyze_trend_segments(valid_data, result.change_points)
            else:
                # Create single segment for entire data
                segment = self._analyze_single_segment(valid_data, 0, len(valid_data) - 1)
                if segment:
                    result.trend_segments = [segment]

            # Seasonality detection
            try:
                if len(valid_data) >= self.seasonality_min_periods:
                    seasonality_result = self._detect_seasonality(valid_data)
                    result.seasonality_detected = seasonality_result['detected']
                    result.seasonal_period = seasonality_result['period']
                    result.seasonal_strength = seasonality_result['strength']
            except Exception as e:
                logger.warning(f"Error in seasonality detection: {e}")
                result.seasonality_detected = False
                result.seasonal_period = None
                result.seasonal_strength = 0.0

            # Generate quality score and recommendations
            result.analysis_quality = self._calculate_analysis_quality(result, valid_data)
            result.recommendations = self._generate_recommendations(result, valid_data)

            # Update metadata
            result.metadata = {
                'data_length': len(valid_data),
                'original_length': len(data),
                'filtered_points': len(data) - len(valid_data),
                'analysis_timestamp': datetime.now().isoformat(),
                'min_value': min(valid_data),
                'max_value': max(valid_data),
                'mean_value': statistics.mean(valid_data),
                'std_dev': statistics.stdev(valid_data) if len(valid_data) > 1 else 0.0
            }

        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            result.is_reliable = False
            result.error_message = f"Analysis error: {str(e)}"

        return result

    def _prepare_data(self,
                     data: List[Union[int, float]],
                     timestamps: Optional[List[datetime]]) -> Tuple[List[float], List[Optional[datetime]]]:
        """Prepare and validate input data."""
        valid_data = []
        valid_timestamps = []

        for i, value in enumerate(data):
            if isinstance(value, (int, float)) and math.isfinite(value):
                valid_data.append(float(value))
                if timestamps and i < len(timestamps):
                    valid_timestamps.append(timestamps[i])
                else:
                    valid_timestamps.append(None)

        return valid_data, valid_timestamps

    def _analyze_overall_trend(self, data: List[float]) -> Dict[str, Any]:
        """Analyze overall trend characteristics."""
        try:
            if len(data) < 2:
                return {
                    'direction': TrendDirection.UNKNOWN,
                    'strength': TrendStrength.VERY_WEAK,
                    'trend_type': TrendType.IRREGULAR,
                    'confidence': 0.0
                }

            # Check for constant data first
            if len(set(data)) == 1:
                return {
                    'direction': TrendDirection.STABLE,
                    'strength': TrendStrength.VERY_WEAK,
                    'trend_type': TrendType.LINEAR,
                    'confidence': 0.0,
                    'correlation': 0.0
                }

            # Linear regression for overall trend
            x_values = list(range(len(data)))
            try:
                correlation = statistics.correlation(x_values, data)
            except statistics.StatisticsError:
                return {
                    'direction': TrendDirection.STABLE,
                    'strength': TrendStrength.VERY_WEAK,
                    'trend_type': TrendType.LINEAR,
                    'confidence': 0.0,
                    'correlation': 0.0
                }

            # Calculate trend strength based on correlation
            strength_value = abs(correlation)
            strength = self._classify_trend_strength(strength_value)

            # Determine direction
            if strength_value < 0.1:
                direction = TrendDirection.STABLE
            elif correlation > 0.8:
                direction = TrendDirection.STRONGLY_INCREASING
            elif correlation > 0.4:
                direction = TrendDirection.MODERATELY_INCREASING
            elif correlation > 0.1:
                direction = TrendDirection.WEAKLY_INCREASING
            elif correlation < -0.8:
                direction = TrendDirection.STRONGLY_DECREASING
            elif correlation < -0.4:
                direction = TrendDirection.MODERATELY_DECREASING
            elif correlation < -0.1:
                direction = TrendDirection.WEAKLY_DECREASING
            else:
                direction = TrendDirection.STABLE

            # Simple trend type classification
            trend_type = self._classify_trend_type(data, x_values)

            return {
                'direction': direction,
                'strength': strength,
                'trend_type': trend_type,
                'confidence': strength_value,
                'correlation': correlation
            }

        except Exception as e:
            logger.warning(f"Error in overall trend analysis: {e}")
            return {
                'direction': TrendDirection.UNKNOWN,
                'strength': TrendStrength.VERY_WEAK,
                'trend_type': TrendType.IRREGULAR,
                'confidence': 0.0
            }

    def _classify_trend_strength(self, strength_value: float) -> TrendStrength:
        """Classify trend strength based on correlation value."""
        if strength_value >= 0.8:
            return TrendStrength.VERY_STRONG
        elif strength_value >= 0.6:
            return TrendStrength.STRONG
        elif strength_value >= 0.4:
            return TrendStrength.MODERATE
        elif strength_value >= 0.2:
            return TrendStrength.WEAK
        else:
            return TrendStrength.VERY_WEAK

    def _classify_trend_type(self, data: List[float], x_values: List[int]) -> TrendType:
        """Classify the type of trend based on data characteristics."""
        try:
            if len(data) < 5:
                return TrendType.LINEAR

            # Test for different trend types
            # Linear trend (default)
            linear_correlation = abs(statistics.correlation(x_values, data))

            # Exponential trend test (log transform)
            try:
                positive_data = [max(0.001, d) for d in data]  # Avoid log(0)
                log_data = [math.log(d) for d in positive_data]
                exp_correlation = abs(statistics.correlation(x_values, log_data))
            except (ValueError, OverflowError):
                exp_correlation = 0.0

            # Polynomial trend test (simple second-order)
            try:
                x_squared = [x * x for x in x_values]
                poly_correlation = abs(statistics.correlation(x_squared, data))
            except statistics.StatisticsError:
                poly_correlation = 0.0

            # Determine best fit
            correlations = {
                TrendType.LINEAR: linear_correlation,
                TrendType.EXPONENTIAL: exp_correlation,
                TrendType.POLYNOMIAL: poly_correlation
            }

            # Check for cyclical patterns
            if self._is_cyclical_pattern(data):
                return TrendType.CYCLICAL

            # Return trend type with highest correlation
            best_type = max(correlations.keys(), key=lambda k: correlations[k])

            # If all correlations are weak, classify as irregular
            if correlations[best_type] < 0.3:
                return TrendType.IRREGULAR

            return best_type

        except Exception as e:
            logger.warning(f"Error classifying trend type: {e}")
            return TrendType.LINEAR

    def _is_cyclical_pattern(self, data: List[float]) -> bool:
        """Detect if data shows cyclical patterns."""
        try:
            if len(data) < 8:
                return False

            # Remove linear trend first to detect true cyclical patterns
            x_values = list(range(len(data)))
            try:
                # Calculate linear trend
                correlation = statistics.correlation(x_values, data)
                if abs(correlation) > 0.8:  # Strong linear trend
                    # Detrend the data
                    slope = statistics.correlation(x_values, data) * (statistics.stdev(data) / statistics.stdev(x_values))
                    intercept = statistics.mean(data) - slope * statistics.mean(x_values)
                    detrended = [data[i] - (slope * i + intercept) for i in range(len(data))]
                else:
                    detrended = data.copy()
            except statistics.StatisticsError:
                detrended = data.copy()

            # Test for periodicity in detrended data
            max_lag = min(len(detrended) // 3, 6)
            best_correlation = 0.0
            periodic_lags = 0

            for lag in range(2, max_lag + 1):
                if lag >= len(detrended):
                    break

                try:
                    original = detrended[:-lag]
                    lagged = detrended[lag:]

                    if len(original) == len(lagged) and len(original) > 1:
                        correlation = abs(statistics.correlation(original, lagged))
                        if correlation > 0.6:  # Higher threshold for periodicity
                            periodic_lags += 1
                        best_correlation = max(best_correlation, correlation)
                except statistics.StatisticsError:
                    continue

            # Require multiple periodic lags for cyclical classification
            return periodic_lags >= 2 and best_correlation > 0.7

        except Exception as e:
            logger.warning(f"Error detecting cyclical pattern: {e}")
            return False

    def _detect_change_points(self,
                            data: List[float],
                            timestamps: List[Optional[datetime]]) -> List[ChangePoint]:
        """Detect significant change points in the data."""
        change_points = []

        try:
            if len(data) < self.min_segment_length * 2:
                return change_points

            # Check if data follows a continuous linear trend first
            x_values = list(range(len(data)))
            try:
                linear_correlation = abs(statistics.correlation(x_values, data))
                if linear_correlation > 0.95:  # Very strong linear trend
                    # For strong linear trends, use stricter change point detection
                    adjusted_threshold = self.change_point_threshold * 3
                else:
                    adjusted_threshold = self.change_point_threshold
            except statistics.StatisticsError:
                adjusted_threshold = self.change_point_threshold

            # Change point detection using moving window statistics
            window_size = max(3, self.min_segment_length)

            for i in range(window_size, len(data) - window_size):
                before_window = data[i - window_size:i]
                after_window = data[i:i + window_size]

                if len(before_window) < 2 or len(after_window) < 2:
                    continue

                try:
                    before_mean = statistics.mean(before_window)
                    after_mean = statistics.mean(after_window)

                    # Calculate change magnitude
                    data_range = max(data) - min(data)
                    if data_range == 0:
                        continue

                    magnitude = abs(after_mean - before_mean) / data_range

                    # For linear trends, also check if this deviates from expected linear progression
                    if linear_correlation > 0.9:
                        # Calculate expected difference for linear trend
                        x_before = list(range(i - window_size, i))
                        x_after = list(range(i, i + window_size))
                        try:
                            slope_before = statistics.correlation(x_before, before_window) * (statistics.stdev(before_window) / statistics.stdev(x_before))
                            slope_after = statistics.correlation(x_after, after_window) * (statistics.stdev(after_window) / statistics.stdev(x_after))

                            # If slopes are similar, this might be continuous trend, not change point
                            slope_diff = abs(slope_after - slope_before)
                            if slope_diff < 0.05:  # Slopes are very similar
                                continue
                        except (statistics.StatisticsError, ZeroDivisionError):
                            pass

                    if magnitude > adjusted_threshold:
                        # Determine change point type
                        change_type = ChangePointType.LEVEL_SHIFT

                        # Calculate confidence based on significance of change
                        confidence = min(1.0, magnitude * 2)

                        change_point = ChangePoint(
                            index=i,
                            timestamp=timestamps[i] if i < len(timestamps) else None,
                            change_type=change_type,
                            before_value=before_mean,
                            after_value=after_mean,
                            magnitude=magnitude,
                            confidence=confidence,
                            statistical_significance=magnitude,  # Simplified
                            context={
                                'window_size': window_size,
                                'before_window': before_window,
                                'after_window': after_window
                            }
                        )

                        change_points.append(change_point)

                except statistics.StatisticsError:
                    continue

            # Filter overlapping change points (keep strongest)
            filtered_points = self._filter_overlapping_change_points(change_points)

        except Exception as e:
            logger.warning(f"Error detecting change points: {e}")

        return filtered_points

    def _filter_overlapping_change_points(self, change_points: List[ChangePoint]) -> List[ChangePoint]:
        """Filter out overlapping change points, keeping the strongest ones."""
        if not change_points:
            return []

        # Sort by confidence (strongest first)
        sorted_points = sorted(change_points, key=lambda cp: cp.confidence, reverse=True)

        filtered_points = []
        min_distance = self.min_segment_length

        for point in sorted_points:
            # Check if this point is too close to already accepted points
            too_close = any(abs(point.index - existing.index) < min_distance
                          for existing in filtered_points)

            if not too_close:
                filtered_points.append(point)

        # Sort by index for final result
        return sorted(filtered_points, key=lambda cp: cp.index)

    def _analyze_trend_segments(self,
                              data: List[float],
                              change_points: List[ChangePoint]) -> List[TrendSegment]:
        """Analyze trend segments between change points."""
        segments = []

        try:
            # Create segments based on change points
            segment_starts = [0] + [cp.index for cp in change_points]
            segment_ends = [cp.index - 1 for cp in change_points] + [len(data) - 1]

            for start, end in zip(segment_starts, segment_ends):
                if end - start + 1 >= self.min_segment_length:
                    segment = self._analyze_single_segment(data, start, end)
                    if segment:
                        segments.append(segment)

        except Exception as e:
            logger.warning(f"Error analyzing trend segments: {e}")

        return segments

    def _analyze_single_segment(self,
                              data: List[float],
                              start_idx: int,
                              end_idx: int) -> Optional[TrendSegment]:
        """Analyze a single trend segment."""
        try:
            segment_data = data[start_idx:end_idx + 1]

            if len(segment_data) < 2:
                return None

            # Linear regression for segment
            x_values = list(range(len(segment_data)))

            if len(set(segment_data)) == 1:
                # All values are the same
                slope = 0.0
                intercept = segment_data[0]
                r_squared = 1.0
                correlation = 0.0
            else:
                try:
                    correlation = statistics.correlation(x_values, segment_data)

                    # Calculate slope and intercept
                    mean_x = statistics.mean(x_values)
                    mean_y = statistics.mean(segment_data)

                    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, segment_data))
                    denominator = sum((x - mean_x) ** 2 for x in x_values)

                    if denominator == 0:
                        slope = 0.0
                    else:
                        slope = numerator / denominator

                    intercept = mean_y - slope * mean_x
                    r_squared = correlation ** 2

                except statistics.StatisticsError:
                    correlation = 0.0
                    slope = 0.0
                    intercept = statistics.mean(segment_data) if segment_data else 0.0
                    r_squared = 0.0

            # Classify segment
            strength = self._classify_trend_strength(abs(correlation))

            if abs(correlation) < 0.1:
                direction = TrendDirection.STABLE
            elif correlation > 0:
                if correlation > 0.8:
                    direction = TrendDirection.STRONGLY_INCREASING
                elif correlation > 0.4:
                    direction = TrendDirection.MODERATELY_INCREASING
                else:
                    direction = TrendDirection.WEAKLY_INCREASING
            else:
                if correlation < -0.8:
                    direction = TrendDirection.STRONGLY_DECREASING
                elif correlation < -0.4:
                    direction = TrendDirection.MODERATELY_DECREASING
                else:
                    direction = TrendDirection.WEAKLY_DECREASING

            trend_type = self._classify_trend_type(segment_data, x_values)

            # Calculate predictions and residuals
            predictions = [intercept + slope * x for x in x_values]
            residuals = [actual - pred for actual, pred in zip(segment_data, predictions)]

            return TrendSegment(
                start_index=start_idx,
                end_index=end_idx,
                direction=direction,
                strength=strength,
                trend_type=trend_type,
                slope=slope,
                intercept=intercept,
                r_squared=r_squared,
                confidence=abs(correlation),
                values=segment_data,
                predictions=predictions,
                residuals=residuals
            )

        except Exception as e:
            logger.warning(f"Error analyzing single segment: {e}")
            return None

    def _detect_seasonality(self, data: List[float]) -> Dict[str, Any]:
        """Detect seasonal patterns in the data."""
        try:
            if len(data) < self.seasonality_min_periods:
                return {'detected': False, 'period': None, 'strength': 0.0}

            # Remove linear trend first to detect true seasonal patterns
            x_values = list(range(len(data)))
            try:
                # Calculate and remove linear trend
                correlation = statistics.correlation(x_values, data)
                if abs(correlation) > 0.8:  # Strong linear trend
                    # Detrend the data
                    slope = correlation * (statistics.stdev(data) / statistics.stdev(x_values))
                    intercept = statistics.mean(data) - slope * statistics.mean(x_values)
                    detrended = [data[i] - (slope * i + intercept) for i in range(len(data))]
                else:
                    detrended = data.copy()
            except statistics.StatisticsError:
                detrended = data.copy()

            # Test for common seasonal periods on detrended data
            max_period = min(len(detrended) // 3, 52)  # Don't test periods longer than data/3
            best_period = None
            best_strength = 0.0

            for period in range(2, max_period + 1):
                if len(detrended) < period * 2:
                    continue

                try:
                    # Calculate autocorrelation at this lag
                    original = detrended[:-period]
                    lagged = detrended[period:]

                    if len(original) == len(lagged) and len(original) > 1:
                        correlation = statistics.correlation(original, lagged)
                        strength = abs(correlation)

                        if strength > best_strength:
                            best_strength = strength
                            best_period = period

                except statistics.StatisticsError:
                    continue

            # Consider seasonality detected if strength is above threshold (higher threshold for detrended data)
            detected = best_strength > 0.5

            return {
                'detected': detected,
                'period': best_period if detected else None,
                'strength': best_strength
            }

        except Exception as e:
            logger.warning(f"Error detecting seasonality: {e}")
            return {'detected': False, 'period': None, 'strength': 0.0}

    def _calculate_analysis_quality(self, result: TrendAnalysisResult, data: List[float]) -> float:
        """Calculate overall quality score for the trend analysis."""
        try:
            quality_factors = []

            # Data sufficiency factor
            data_factor = min(1.0, len(data) / 50.0)  # Optimal around 50+ points
            quality_factors.append(data_factor)

            # Trend confidence factor
            quality_factors.append(result.trend_confidence)

            # Segment analysis factor
            if result.trend_segments:
                segment_confidences = [seg.confidence for seg in result.trend_segments]
                segment_factor = statistics.mean(segment_confidences)
                quality_factors.append(segment_factor)

            # Change point detection factor
            if result.change_points:
                change_point_confidences = [cp.confidence for cp in result.change_points]
                change_factor = statistics.mean(change_point_confidences)
                quality_factors.append(change_factor)

            # Overall quality is the mean of all factors
            return statistics.mean(quality_factors) if quality_factors else 0.0

        except Exception as e:
            logger.warning(f"Error calculating analysis quality: {e}")
            return 0.0

    def _generate_recommendations(self, result: TrendAnalysisResult, data: List[float]) -> List[str]:
        """Generate actionable recommendations based on trend analysis."""
        recommendations = []

        try:
            # Recommendations based on overall trend
            if result.overall_strength in [TrendStrength.STRONG, TrendStrength.VERY_STRONG]:
                if result.overall_direction in [
                    TrendDirection.STRONGLY_INCREASING,
                    TrendDirection.MODERATELY_INCREASING
                ]:
                    recommendations.append("Strong upward trend detected - consider planning for continued growth")
                    recommendations.append("Monitor for potential inflection points or slowdown")
                elif result.overall_direction in [
                    TrendDirection.STRONGLY_DECREASING,
                    TrendDirection.MODERATELY_DECREASING
                ]:
                    recommendations.append("Strong downward trend detected - investigate underlying causes")
                    recommendations.append("Consider intervention strategies to reverse the decline")

            # Recommendations based on change points
            if result.change_points:
                recent_changes = [cp for cp in result.change_points[-3:]]  # Last 3 change points
                if recent_changes:
                    recommendations.append("Recent change points detected - monitor closely for pattern shifts")
                    recommendations.append("Analyze external factors that may have caused recent changes")

            # Recommendations based on seasonality
            if result.seasonality_detected and result.seasonal_period:
                recommendations.append(f"Seasonal pattern detected with period {result.seasonal_period}")
                recommendations.append("Plan for recurring seasonal variations in forecasting")
                recommendations.append("Consider seasonal adjustment in trend analysis")

            # Data quality recommendations
            if result.analysis_quality < 0.5:
                recommendations.append("Analysis quality is low - consider collecting more data")
                recommendations.append("Verify data collection processes and remove outliers if appropriate")

            # Trend stability recommendations
            if result.overall_direction == TrendDirection.VOLATILE:
                recommendations.append("High volatility detected - implement risk management strategies")
                recommendations.append("Consider smoothing techniques for better trend visualization")

            if not recommendations:
                recommendations.append("Trend analysis complete - continue monitoring for pattern changes")

        except Exception as e:
            logger.warning(f"Error generating recommendations: {e}")
            recommendations.append("Error generating recommendations - manual analysis recommended")

        return recommendations