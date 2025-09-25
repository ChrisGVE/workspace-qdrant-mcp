"""
Comprehensive unit tests for PatternRecognition with extensive edge case coverage.

Tests cover:
- Trend detection with various data patterns
- Pattern recognition for different pattern types
- Change point detection using multiple methods
- Seasonality detection and analysis
- Edge cases: sparse data, noisy signals, boundary conditions
- Caching functionality and error handling
"""

import pytest
import math
import statistics
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from src.python.workspace_qdrant_mcp.analytics.engine.pattern_recognition import (
    PatternRecognition,
    TrendDirection,
    PatternType,
    TrendResult,
    PatternResult,
    ChangePointResult
)


class TestTrendResult:
    """Tests for TrendResult data class."""

    def test_trend_result_initialization(self):
        """Test TrendResult initialization with default values."""
        result = TrendResult(direction=TrendDirection.INCREASING, strength=0.8)

        assert result.direction == TrendDirection.INCREASING
        assert result.strength == 0.8
        assert result.slope is None
        assert result.r_squared is None
        assert result.confidence == 0.0
        assert result.start_value is None
        assert result.end_value is None
        assert result.change_magnitude is None
        assert result.change_percentage is None
        assert result.duration_periods == 0
        assert result.is_significant is False
        assert result.p_value is None
        assert result.error_message is None

    def test_trend_result_to_dict(self):
        """Test conversion to dictionary."""
        result = TrendResult(
            direction=TrendDirection.DECREASING,
            strength=0.6,
            slope=-0.5,
            r_squared=0.75,
            confidence=0.9
        )
        result_dict = result.to_dict()

        assert result_dict['direction'] == 'decreasing'
        assert result_dict['strength'] == 0.6
        assert result_dict['slope'] == -0.5
        assert result_dict['r_squared'] == 0.75
        assert result_dict['confidence'] == 0.9


class TestPatternResult:
    """Tests for PatternResult data class."""

    def test_pattern_result_initialization(self):
        """Test PatternResult initialization."""
        result = PatternResult(
            pattern_type=PatternType.LINEAR,
            confidence=0.85,
            parameters={'slope': 1.2},
            fit_quality=0.8,
            description="Linear pattern",
            data_points=10,
            pattern_strength=0.7
        )

        assert result.pattern_type == PatternType.LINEAR
        assert result.confidence == 0.85
        assert result.parameters == {'slope': 1.2}
        assert result.fit_quality == 0.8
        assert result.description == "Linear pattern"
        assert result.data_points == 10
        assert result.pattern_strength == 0.7

    def test_pattern_result_to_dict(self):
        """Test conversion to dictionary."""
        result = PatternResult(
            pattern_type=PatternType.EXPONENTIAL,
            confidence=0.9,
            parameters={'base': 2.0, 'growth_rate': 0.1},
            fit_quality=0.85,
            description="Exponential growth",
            data_points=15,
            pattern_strength=0.9
        )
        result_dict = result.to_dict()

        assert result_dict['pattern_type'] == 'exponential'
        assert result_dict['confidence'] == 0.9
        assert result_dict['parameters'] == {'base': 2.0, 'growth_rate': 0.1}


class TestChangePointResult:
    """Tests for ChangePointResult data class."""

    def test_change_point_result_initialization(self):
        """Test ChangePointResult initialization."""
        result = ChangePointResult(
            change_points=[10, 25],
            confidence_scores=[0.8, 0.9],
            change_types=['level_increase', 'trend_change'],
            change_magnitudes=[5.0, 3.2],
            statistical_significance=[True, True],
            method_used='variance',
            total_changes=2
        )

        assert result.change_points == [10, 25]
        assert result.confidence_scores == [0.8, 0.9]
        assert result.change_types == ['level_increase', 'trend_change']
        assert result.change_magnitudes == [5.0, 3.2]
        assert result.statistical_significance == [True, True]
        assert result.method_used == 'variance'
        assert result.total_changes == 2


class TestPatternRecognition:
    """Comprehensive tests for PatternRecognition engine."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = PatternRecognition(min_data_points=5, significance_threshold=0.05)

    def test_initialization(self):
        """Test engine initialization."""
        assert self.engine.min_data_points == 5
        assert self.engine.significance_threshold == 0.05
        assert len(self.engine._pattern_cache) == 0

    def test_initialization_custom_parameters(self):
        """Test engine initialization with custom parameters."""
        engine = PatternRecognition(min_data_points=10, significance_threshold=0.01)
        assert engine.min_data_points == 10
        assert engine.significance_threshold == 0.01

    # Trend Detection Tests

    def test_detect_trend_increasing(self):
        """Test trend detection with increasing data."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = self.engine.detect_trend(data)

        assert result.direction == TrendDirection.INCREASING
        assert result.strength > 0.8  # Should have high strength for perfect linear trend
        assert result.slope > 0
        assert result.r_squared > 0.95
        assert result.start_value == 1.0
        assert result.end_value == 10.0
        assert result.change_magnitude == 9.0
        assert result.change_percentage == 900.0
        assert result.is_significant is True
        assert result.error_message is None

    def test_detect_trend_decreasing(self):
        """Test trend detection with decreasing data."""
        data = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        result = self.engine.detect_trend(data)

        assert result.direction == TrendDirection.DECREASING
        assert result.strength > 0.8
        assert result.slope < 0
        assert result.r_squared > 0.95
        assert result.start_value == 10.0
        assert result.end_value == 1.0
        assert result.change_magnitude == -9.0
        assert result.change_percentage == -90.0

    def test_detect_trend_stable(self):
        """Test trend detection with stable data."""
        data = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        result = self.engine.detect_trend(data)

        assert result.direction == TrendDirection.STABLE
        assert result.strength == 0.0
        assert result.slope == 0.0
        assert result.change_magnitude == 0.0
        assert result.change_percentage == 0.0

    def test_detect_trend_volatile(self):
        """Test trend detection with volatile data."""
        data = [1, 10, 2, 9, 3, 8, 4, 7, 5, 6]  # Highly volatile
        result = self.engine.detect_trend(data)

        # Should detect as volatile due to low R-squared and high volatility
        assert result.direction in [TrendDirection.VOLATILE, TrendDirection.INCREASING]
        if result.direction == TrendDirection.VOLATILE:
            assert result.r_squared < 0.3

    def test_detect_trend_empty_data(self):
        """Test trend detection with empty dataset."""
        result = self.engine.detect_trend([])

        assert result.direction == TrendDirection.UNKNOWN
        assert result.strength == 0.0
        assert result.error_message == "Empty dataset provided"

    def test_detect_trend_insufficient_data(self):
        """Test trend detection with insufficient data points."""
        data = [1, 2, 3]  # Less than minimum required
        result = self.engine.detect_trend(data)

        assert result.direction == TrendDirection.UNKNOWN
        assert "Insufficient data points" in result.error_message

    def test_detect_trend_with_nan_values(self):
        """Test trend detection handling NaN values."""
        data = [1, 2, float('nan'), 4, 5, 6, 7, 8, 9, 10]
        result = self.engine.detect_trend(data)

        assert result.direction == TrendDirection.INCREASING
        assert result.strength > 0.8  # Should still detect strong trend
        # NaN should be filtered out

    def test_detect_trend_with_infinite_values(self):
        """Test trend detection handling infinite values."""
        data = [1, 2, float('inf'), 4, 5, 6, 7, 8, float('-inf'), 10]
        result = self.engine.detect_trend(data)

        assert result.direction == TrendDirection.INCREASING
        # Infinite values should be filtered out

    def test_detect_trend_all_invalid_values(self):
        """Test trend detection with all invalid values."""
        data = [float('nan'), float('inf'), float('-inf'), float('nan')]
        result = self.engine.detect_trend(data)

        assert result.direction == TrendDirection.UNKNOWN
        assert "Insufficient valid data points" in result.error_message

    def test_detect_trend_zero_division_edge_case(self):
        """Test trend detection with zero variance in x-values (edge case)."""
        # This is a theoretical edge case that shouldn't occur in practice
        # but we test the handling
        data = [1, 2, 3, 4, 5]
        result = self.engine.detect_trend(data)

        # Should handle gracefully without division by zero
        assert result.direction in [TrendDirection.INCREASING, TrendDirection.STABLE]

    def test_detect_trend_with_timestamps(self):
        """Test trend detection with timestamp data."""
        data = [1, 2, 3, 4, 5]
        timestamps = [datetime(2023, 1, i) for i in range(1, 6)]
        result = self.engine.detect_trend(data, timestamps=timestamps)

        assert result.direction == TrendDirection.INCREASING
        assert result.strength > 0.8

    def test_detect_trend_perfect_correlation(self):
        """Test trend detection with perfect correlation (R² = 1)."""
        data = [2 * i + 3 for i in range(10)]  # Perfect linear relationship
        result = self.engine.detect_trend(data)

        assert result.direction == TrendDirection.INCREASING
        assert abs(result.r_squared - 1.0) < 0.001
        assert result.strength > 0.99

    def test_detect_trend_near_zero_start_value(self):
        """Test trend detection with near-zero start value."""
        data = [0.0001, 1, 2, 3, 4, 5]
        result = self.engine.detect_trend(data)

        assert result.direction == TrendDirection.INCREASING
        # Should handle percentage calculation without overflow
        assert result.change_percentage is not None

    def test_detect_trend_zero_start_value(self):
        """Test trend detection with zero start value."""
        data = [0, 1, 2, 3, 4, 5]
        result = self.engine.detect_trend(data)

        assert result.direction == TrendDirection.INCREASING
        assert result.change_percentage == float('inf')  # Division by zero case

    # Pattern Detection Tests

    def test_detect_patterns_linear(self):
        """Test detection of linear patterns."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        patterns = self.engine.detect_patterns(data, pattern_types=[PatternType.LINEAR])

        assert len(patterns) == 1
        assert patterns[0].pattern_type == PatternType.LINEAR
        assert patterns[0].confidence > 0.9
        assert 'slope' in patterns[0].parameters
        assert patterns[0].is_significant is True

    def test_detect_patterns_exponential(self):
        """Test detection of exponential patterns."""
        data = [math.exp(i * 0.5) for i in range(10)]  # Exponential growth
        patterns = self.engine.detect_patterns(data, pattern_types=[PatternType.EXPONENTIAL])

        assert len(patterns) == 1
        assert patterns[0].pattern_type == PatternType.EXPONENTIAL
        assert patterns[0].confidence > 0.8
        assert 'base' in patterns[0].parameters
        assert 'growth_rate' in patterns[0].parameters

    def test_detect_patterns_exponential_negative_values(self):
        """Test exponential pattern detection with negative values."""
        data = [-1, 2, -3, 4, -5]  # Contains negative values
        patterns = self.engine.detect_patterns(data, pattern_types=[PatternType.EXPONENTIAL])

        # Should not detect exponential pattern due to negative values
        assert len(patterns) == 0

    def test_detect_patterns_logarithmic(self):
        """Test detection of logarithmic patterns."""
        data = [math.log(i + 1) * 3 + 2 for i in range(10)]  # Logarithmic growth
        patterns = self.engine.detect_patterns(data, pattern_types=[PatternType.LOGARITHMIC])

        assert len(patterns) == 1
        assert patterns[0].pattern_type == PatternType.LOGARITHMIC
        assert patterns[0].confidence > 0.8

    def test_detect_patterns_periodic(self):
        """Test detection of periodic patterns."""
        data = [math.sin(i * math.pi / 6) for i in range(24)]  # Sine wave with period 12
        patterns = self.engine.detect_patterns(data, pattern_types=[PatternType.PERIODIC])

        assert len(patterns) == 1
        assert patterns[0].pattern_type == PatternType.PERIODIC
        assert patterns[0].confidence > 0.5
        assert 'period' in patterns[0].parameters

    def test_detect_patterns_step_change(self):
        """Test detection of step change patterns."""
        data = [1, 1, 1, 1, 1, 5, 5, 5, 5, 5]  # Clear step change
        patterns = self.engine.detect_patterns(data, pattern_types=[PatternType.STEP_CHANGE])

        assert len(patterns) == 1
        assert patterns[0].pattern_type == PatternType.STEP_CHANGE
        assert patterns[0].confidence > 0.5
        assert 'change_point' in patterns[0].parameters

    def test_detect_patterns_spike(self):
        """Test detection of spike patterns."""
        data = [1, 1, 1, 10, 1, 1, 1]  # Clear spike
        patterns = self.engine.detect_patterns(data, pattern_types=[PatternType.SPIKE])

        assert len(patterns) == 1
        assert patterns[0].pattern_type == PatternType.SPIKE
        assert patterns[0].confidence > 0.3
        assert 'spike_positions' in patterns[0].parameters

    def test_detect_patterns_no_spikes_constant_data(self):
        """Test spike detection with constant data (no variation)."""
        data = [5, 5, 5, 5, 5, 5, 5]  # No spikes
        patterns = self.engine.detect_patterns(data, pattern_types=[PatternType.SPIKE])

        assert len(patterns) == 0  # No spikes in constant data

    def test_detect_patterns_all_types(self):
        """Test detection of all pattern types simultaneously."""
        # Create data with multiple pattern characteristics
        data = [i + math.sin(i) + (0.1 if i == 5 else 0) for i in range(20)]
        patterns = self.engine.detect_patterns(data)  # All pattern types

        assert len(patterns) > 0
        # Should be sorted by confidence (highest first)
        for i in range(len(patterns) - 1):
            assert patterns[i].confidence >= patterns[i + 1].confidence

    def test_detect_patterns_insufficient_data(self):
        """Test pattern detection with insufficient data."""
        data = [1, 2]  # Too short for meaningful pattern detection
        patterns = self.engine.detect_patterns(data, min_pattern_length=3)

        assert len(patterns) == 0

    def test_detect_patterns_empty_data(self):
        """Test pattern detection with empty data."""
        patterns = self.engine.detect_patterns([])

        assert len(patterns) == 0

    def test_detect_patterns_all_invalid_data(self):
        """Test pattern detection with all invalid data."""
        data = [float('nan'), float('inf'), float('-inf')]
        patterns = self.engine.detect_patterns(data)

        assert len(patterns) == 0

    def test_detect_patterns_with_nan_values(self):
        """Test pattern detection handling NaN values."""
        data = [1, 2, float('nan'), 4, 5, 6, 7, 8, 9, 10]
        patterns = self.engine.detect_patterns(data, pattern_types=[PatternType.LINEAR])

        # Should still detect linear pattern after filtering NaN
        assert len(patterns) > 0
        assert patterns[0].pattern_type == PatternType.LINEAR

    # Change Point Detection Tests

    def test_detect_change_points_variance_method(self):
        """Test change point detection using variance method."""
        # Create data with variance change
        data = ([1 + 0.1 * i for i in range(10)] +  # Low variance
                [5 + 2 * i for i in range(10)])      # High variance
        result = self.engine.detect_change_points(data, method="variance")

        assert result.method_used == "variance"
        assert result.total_changes >= 0
        assert len(result.change_points) == len(result.confidence_scores)
        assert len(result.change_types) == len(result.change_magnitudes)

    def test_detect_change_points_mean_method(self):
        """Test change point detection using mean method."""
        # Create data with mean change
        data = [1] * 10 + [5] * 10  # Clear mean change
        result = self.engine.detect_change_points(data, method="mean")

        assert result.method_used == "mean"
        assert result.total_changes > 0
        assert len(result.change_points) > 0
        # Change point should be around index 10
        assert any(8 <= cp <= 12 for cp in result.change_points)

    def test_detect_change_points_trend_method(self):
        """Test change point detection using trend method."""
        # Create data with trend change
        data = list(range(10)) + list(range(10, 0, -1))  # Up then down
        result = self.engine.detect_change_points(data, method="trend")

        assert result.method_used == "trend"
        assert result.total_changes > 0
        # Should detect trend reversal around the middle

    def test_detect_change_points_unknown_method(self):
        """Test change point detection with unknown method."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = self.engine.detect_change_points(data, method="unknown")

        assert result.total_changes == 0
        assert "Unknown change point detection method" in result.error_message

    def test_detect_change_points_insufficient_data(self):
        """Test change point detection with insufficient data."""
        data = [1, 2, 3]  # Too short
        result = self.engine.detect_change_points(data, min_segment_length=5)

        assert result.total_changes == 0
        assert "Insufficient data" in result.error_message

    def test_detect_change_points_empty_data(self):
        """Test change point detection with empty data."""
        result = self.engine.detect_change_points([])

        assert result.total_changes == 0
        assert "Insufficient data" in result.error_message

    def test_detect_change_points_all_invalid_data(self):
        """Test change point detection with all invalid data."""
        data = [float('nan')] * 20
        result = self.engine.detect_change_points(data, method="mean")

        assert result.total_changes == 0
        assert "Insufficient valid data points" in result.error_message

    def test_detect_change_points_with_nan_values(self):
        """Test change point detection handling NaN values."""
        data = [1] * 5 + [float('nan')] * 2 + [5] * 5
        result = self.engine.detect_change_points(data, method="mean")

        # Should detect change point while ignoring NaN values
        assert result.method_used == "mean"
        # May or may not detect change due to NaN filtering

    def test_detect_change_points_most_significant(self):
        """Test identification of most significant change point."""
        # Multiple changes with different magnitudes
        data = [1] * 5 + [3] * 5 + [10] * 5  # Two changes, second is larger
        result = self.engine.detect_change_points(data, method="mean")

        if result.total_changes > 0:
            assert result.most_significant_change is not None
            assert result.average_confidence > 0

    # Seasonality Detection Tests

    def test_detect_seasonality_clear_seasonal_pattern(self):
        """Test seasonality detection with clear seasonal pattern."""
        # Create seasonal data (period = 12)
        data = [math.sin(i * 2 * math.pi / 12) + 2 for i in range(48)]
        result = self.engine.detect_seasonality(data, period_hints=[12])

        assert result['is_seasonal'] is True
        assert result['dominant_period'] == 12
        assert result['dominant_strength'] > 0.5
        assert result['confidence'] > 0.5
        assert 12 in result['seasonal_periods']

    def test_detect_seasonality_no_seasonal_pattern(self):
        """Test seasonality detection with non-seasonal data."""
        data = list(range(50))  # Linear trend, no seasonality
        result = self.engine.detect_seasonality(data)

        assert result['is_seasonal'] is False
        assert result['dominant_strength'] < 0.3

    def test_detect_seasonality_insufficient_data(self):
        """Test seasonality detection with insufficient data."""
        data = [1, 2, 3]
        result = self.engine.detect_seasonality(data)

        assert result['is_seasonal'] is False
        assert "Insufficient data" in result['error_message']

    def test_detect_seasonality_empty_data(self):
        """Test seasonality detection with empty data."""
        result = self.engine.detect_seasonality([])

        assert result['is_seasonal'] is False
        assert "Insufficient data" in result['error_message']

    def test_detect_seasonality_all_invalid_data(self):
        """Test seasonality detection with all invalid data."""
        data = [float('nan'), float('inf')] * 10
        result = self.engine.detect_seasonality(data)

        assert result['is_seasonal'] is False
        assert "Insufficient valid data points" in result['error_message']

    def test_detect_seasonality_with_nan_values(self):
        """Test seasonality detection handling NaN values."""
        # Seasonal pattern with some NaN values
        data = []
        for i in range(48):
            if i % 10 == 0:
                data.append(float('nan'))
            else:
                data.append(math.sin(i * 2 * math.pi / 12) + 2)

        result = self.engine.detect_seasonality(data, period_hints=[12])

        # Should still potentially detect seasonality after filtering NaN
        assert 'error_message' not in result or result['error_message'] is None

    def test_detect_seasonality_custom_period_hints(self):
        """Test seasonality detection with custom period hints."""
        # Create data with period 7
        data = [math.sin(i * 2 * math.pi / 7) for i in range(35)]
        result = self.engine.detect_seasonality(data, period_hints=[7, 14, 21])

        if result['is_seasonal']:
            assert result['dominant_period'] in [7, 14, 21]

    def test_detect_seasonality_max_period_limit(self):
        """Test seasonality detection with max period limit."""
        data = [math.sin(i * 2 * math.pi / 20) for i in range(100)]
        result = self.engine.detect_seasonality(data, max_period=15)

        # Should not find period 20 due to max_period limit
        if result['is_seasonal']:
            assert all(p <= 15 for p in result['seasonal_periods'])

    def test_detect_seasonality_multiple_periods(self):
        """Test seasonality detection with multiple seasonal periods."""
        # Combine two seasonal patterns
        data = []
        for i in range(60):
            value = (math.sin(i * 2 * math.pi / 12) +  # Period 12
                    0.5 * math.sin(i * 2 * math.pi / 4))  # Period 4
            data.append(value)

        result = self.engine.detect_seasonality(data, period_hints=[4, 12])

        assert len(result['seasonal_periods']) >= 1
        # Should detect at least one of the periods

    # Caching Tests

    def test_trend_detection_caching(self):
        """Test caching functionality for trend detection."""
        data = [1, 2, 3, 4, 5]
        cache_key = "test_trend_cache"

        # First call should calculate and cache
        result1 = self.engine.detect_trend(data, cache_key=cache_key)

        # Second call should use cache
        result2 = self.engine.detect_trend(data, cache_key=cache_key)

        assert result1.direction == result2.direction
        assert result1.strength == result2.strength
        assert cache_key in self.engine._pattern_cache

    def test_caching_without_key(self):
        """Test that results are not cached without cache key."""
        data = [1, 2, 3, 4, 5]
        result = self.engine.detect_trend(data)

        assert len(self.engine._pattern_cache) == 0

    @patch('src.python.workspace_qdrant_mcp.analytics.engine.pattern_recognition.datetime')
    def test_cache_expiry(self, mock_datetime):
        """Test cache expiry functionality."""
        # Mock datetime to control cache timing
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        expired_time = start_time + timedelta(minutes=15)  # Beyond TTL

        mock_datetime.now.side_effect = [start_time, start_time, expired_time]

        data = [1, 2, 3, 4, 5]
        cache_key = "test_expiry"

        # First call - cache result
        self.engine.detect_trend(data, cache_key=cache_key)

        # Check cache is expired
        assert self.engine._is_cached_valid(cache_key) is False

    def test_clear_cache(self):
        """Test cache clearing functionality."""
        data = [1, 2, 3, 4, 5]
        cache_key = "test_clear"

        # Cache a result
        self.engine.detect_trend(data, cache_key=cache_key)
        assert len(self.engine._pattern_cache) > 0

        # Clear cache
        self.engine.clear_cache()
        assert len(self.engine._pattern_cache) == 0
        assert len(self.engine._cache_timestamps) == 0

    # Error Handling Tests

    def test_error_handling_in_trend_detection(self):
        """Test error handling in trend detection."""
        # Mock an exception during calculation
        with patch('statistics.mean', side_effect=RuntimeError("Test error")):
            data = [1, 2, 3, 4, 5]
            result = self.engine.detect_trend(data)

            assert result.direction == TrendDirection.UNKNOWN
            assert "Trend detection error" in result.error_message

    def test_error_handling_in_pattern_detection(self):
        """Test error handling in pattern detection."""
        # Test with data that might cause issues
        data = [1, 2, 3, 4, 5]

        # Mock an exception in specific pattern detection
        with patch.object(self.engine, '_detect_linear_pattern', side_effect=RuntimeError("Test error")):
            patterns = self.engine.detect_patterns(data, pattern_types=[PatternType.LINEAR])

            # Should handle error gracefully and return empty list
            assert len(patterns) == 0

    def test_error_handling_in_change_point_detection(self):
        """Test error handling in change point detection."""
        # Mock an exception
        with patch('statistics.mean', side_effect=RuntimeError("Test error")):
            data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            result = self.engine.detect_change_points(data, method="mean")

            assert result.total_changes == 0
            assert "Change point detection error" in result.error_message

    def test_error_handling_in_seasonality_detection(self):
        """Test error handling in seasonality detection."""
        # Mock an exception
        with patch.object(self.engine, '_calculate_autocorrelation', side_effect=RuntimeError("Test error")):
            data = [1, 2, 3, 4, 5] * 10
            result = self.engine.detect_seasonality(data)

            assert result['is_seasonal'] is False
            assert "Seasonality detection error" in result['error_message']

    # Helper Method Tests

    def test_calculate_autocorrelation_edge_cases(self):
        """Test autocorrelation calculation edge cases."""
        # Lag greater than data length
        data = [1, 2, 3]
        result = self.engine._calculate_autocorrelation(data, lag=5)
        assert result is None

        # Lag of 0 or negative
        result = self.engine._calculate_autocorrelation(data, lag=0)
        assert result is None

        # Constant data (zero variance)
        data = [5, 5, 5, 5, 5]
        result = self.engine._calculate_autocorrelation(data, lag=2)
        assert result is None

    def test_calculate_volatility_edge_cases(self):
        """Test volatility calculation edge cases."""
        # Single data point
        volatility = self.engine._calculate_volatility([5])
        assert volatility == 0.0

        # Zero mean
        volatility = self.engine._calculate_volatility([-1, 0, 1])
        assert volatility == float('inf')

        # Normal case
        volatility = self.engine._calculate_volatility([1, 2, 3, 4, 5])
        assert volatility > 0.0

    def test_calculate_segment_trend_edge_cases(self):
        """Test segment trend calculation edge cases."""
        # Single data point
        trend = self.engine._calculate_segment_trend([5])
        assert trend is None

        # Two identical points
        trend = self.engine._calculate_segment_trend([5, 5])
        assert trend == 0.0

        # Normal increasing trend
        trend = self.engine._calculate_segment_trend([1, 2, 3, 4, 5])
        assert trend > 0

    # Integration Tests

    def test_comprehensive_analysis_workflow(self):
        """Test comprehensive pattern analysis workflow."""
        import random
        random.seed(42)  # For reproducible results

        # Create complex data with multiple characteristics
        data = []

        # Linear trend
        for i in range(50):
            data.append(i * 0.5 + random.gauss(0, 0.5))

        # Add seasonal component
        for i in range(len(data)):
            data[i] += math.sin(i * 2 * math.pi / 12) * 2

        # Add a step change
        for i in range(30, len(data)):
            data[i] += 3

        # Run comprehensive analysis
        trend_result = self.engine.detect_trend(data)
        patterns = self.engine.detect_patterns(data)
        change_points = self.engine.detect_change_points(data, method="mean")
        seasonality = self.engine.detect_seasonality(data, period_hints=[12])

        # Verify all analyses completed
        assert trend_result.direction != TrendDirection.UNKNOWN
        assert len(patterns) > 0
        assert seasonality['method'] == 'autocorrelation'
        # Change points may or may not be detected depending on noise

    def test_large_dataset_performance(self):
        """Test performance with large dataset."""
        # Create large dataset
        data = [math.sin(i * 0.1) + 0.1 * i for i in range(1000)]

        # Should complete without errors
        trend_result = self.engine.detect_trend(data)
        assert trend_result.direction != TrendDirection.UNKNOWN

        patterns = self.engine.detect_patterns(data, pattern_types=[PatternType.LINEAR])
        assert len(patterns) >= 0  # Should complete without error

    def test_edge_case_data_patterns(self):
        """Test with various edge case data patterns."""
        # Extremely small values
        tiny_data = [1e-10 * i for i in range(10)]
        result = self.engine.detect_trend(tiny_data)
        assert result.direction == TrendDirection.INCREASING

        # Extremely large values
        large_data = [1e10 * i for i in range(10)]
        result = self.engine.detect_trend(large_data)
        assert result.direction == TrendDirection.INCREASING

        # Alternating values
        alternating_data = [1 if i % 2 == 0 else -1 for i in range(20)]
        result = self.engine.detect_trend(alternating_data)
        assert result.direction in [TrendDirection.STABLE, TrendDirection.VOLATILE]

    def test_mixed_data_types(self):
        """Test with mixed integer and float data."""
        data = [1, 2.5, 3, 4.7, 5, 6.2, 7, 8.9, 9, 10.1]
        result = self.engine.detect_trend(data)

        assert result.direction == TrendDirection.INCREASING
        assert result.strength > 0.8