"""
Comprehensive unit tests for StatisticalEngine with extensive edge case coverage.

Tests cover:
- Normal statistical calculations
- Edge cases: NaN, infinity, zero division
- Empty datasets and invalid data
- Outlier detection with various methods
- Correlation analysis edge cases
- Moving average calculations
- Percentile calculations
- Caching functionality
"""

import pytest
import math
import statistics
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from src.python.workspace_qdrant_mcp.analytics.engine.statistical_engine import (
    StatisticalEngine,
    StatisticalResult
)


class TestStatisticalResult:
    """Tests for StatisticalResult data class."""

    def test_statistical_result_initialization(self):
        """Test StatisticalResult initialization with default values."""
        result = StatisticalResult()
        assert result.mean is None
        assert result.median is None
        assert result.mode is None
        assert result.std_dev is None
        assert result.variance is None
        assert result.min_value is None
        assert result.max_value is None
        assert result.count == 0
        assert result.sum_value is None
        assert result.percentiles is None
        assert result.is_valid is True
        assert result.error_message is None

    def test_statistical_result_to_dict(self):
        """Test conversion to dictionary."""
        result = StatisticalResult(
            mean=5.0,
            median=4.5,
            count=10,
            is_valid=True
        )
        result_dict = result.to_dict()

        assert result_dict['mean'] == 5.0
        assert result_dict['median'] == 4.5
        assert result_dict['count'] == 10
        assert result_dict['is_valid'] is True
        assert result_dict['error_message'] is None

    def test_statistical_result_invalid_state(self):
        """Test StatisticalResult in invalid state."""
        result = StatisticalResult(
            is_valid=False,
            error_message="Test error"
        )
        assert result.is_valid is False
        assert result.error_message == "Test error"


class TestStatisticalEngine:
    """Comprehensive tests for StatisticalEngine."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = StatisticalEngine(precision_digits=4)

    def test_initialization(self):
        """Test engine initialization."""
        assert self.engine.precision_digits == 4
        assert len(self.engine._stats_cache) == 0
        assert len(self.engine._cache_timestamps) == 0

    def test_initialization_custom_precision(self):
        """Test engine initialization with custom precision."""
        engine = StatisticalEngine(precision_digits=2)
        assert engine.precision_digits == 2

    # Basic Statistics Tests

    def test_calculate_basic_stats_normal_data(self):
        """Test basic statistics with normal dataset."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = self.engine.calculate_basic_stats(data)

        assert result.is_valid is True
        assert result.count == 10
        assert result.mean == 5.5
        assert result.median == 5.5
        assert result.min_value == 1.0
        assert result.max_value == 10.0
        assert result.sum_value == 55.0
        assert abs(result.std_dev - 3.0277) < 0.0001  # Approximate check
        assert result.percentiles is not None
        assert 50 in result.percentiles
        assert result.error_message is None

    def test_calculate_basic_stats_empty_data(self):
        """Test basic statistics with empty dataset."""
        result = self.engine.calculate_basic_stats([])

        assert result.is_valid is False
        assert result.error_message == "Empty dataset provided"
        assert result.count == 0

    def test_calculate_basic_stats_single_value(self):
        """Test basic statistics with single value."""
        data = [42.5]
        result = self.engine.calculate_basic_stats(data)

        assert result.is_valid is True
        assert result.count == 1
        assert result.mean == 42.5
        assert result.median == 42.5
        assert result.mode == 42.5
        assert result.std_dev == 0.0  # Single value has no deviation
        assert result.variance == 0.0
        assert result.min_value == 42.5
        assert result.max_value == 42.5

    def test_calculate_basic_stats_with_nan_values(self):
        """Test handling of NaN values in dataset."""
        data = [1, 2, float('nan'), 4, 5]
        result = self.engine.calculate_basic_stats(data)

        assert result.is_valid is True
        assert result.count == 4  # NaN excluded
        assert result.mean == 3.0
        assert result.median == 3.0

    def test_calculate_basic_stats_with_infinity_values(self):
        """Test handling of infinity values in dataset."""
        data = [1, 2, float('inf'), 4, float('-inf'), 5]
        result = self.engine.calculate_basic_stats(data)

        assert result.is_valid is True
        assert result.count == 4  # Infinities excluded
        assert result.mean == 3.0
        assert result.median == 3.0

    def test_calculate_basic_stats_all_invalid_values(self):
        """Test dataset with only invalid values (NaN, inf)."""
        data = [float('nan'), float('inf'), float('-inf')]
        result = self.engine.calculate_basic_stats(data)

        assert result.is_valid is False
        assert result.error_message == "No valid numerical values found"

    def test_calculate_basic_stats_mixed_types(self):
        """Test dataset with mixed numerical types."""
        data = [1, 2.5, 3, 4.7, 5]
        result = self.engine.calculate_basic_stats(data)

        assert result.is_valid is True
        assert result.count == 5
        assert abs(result.mean - 3.24) < 0.01

    def test_calculate_basic_stats_identical_values(self):
        """Test dataset with identical values (no variance)."""
        data = [5, 5, 5, 5, 5]
        result = self.engine.calculate_basic_stats(data)

        assert result.is_valid is True
        assert result.mean == 5.0
        assert result.median == 5.0
        assert result.mode == 5.0
        assert result.std_dev == 0.0
        assert result.variance == 0.0

    def test_calculate_basic_stats_large_numbers(self):
        """Test with very large numbers."""
        data = [1e10, 2e10, 3e10, 4e10, 5e10]
        result = self.engine.calculate_basic_stats(data)

        assert result.is_valid is True
        assert result.count == 5
        assert abs(result.mean - 3e10) < 1e6  # Allow for floating point precision

    def test_calculate_basic_stats_very_small_numbers(self):
        """Test with very small numbers."""
        data = [1e-10, 2e-10, 3e-10, 4e-10, 5e-10]
        result = self.engine.calculate_basic_stats(data)

        assert result.is_valid is True
        assert result.count == 5
        assert abs(result.mean - 3e-10) < 1e-12

    def test_calculate_basic_stats_precision_rounding(self):
        """Test precision rounding in calculations."""
        engine = StatisticalEngine(precision_digits=2)
        data = [1.23456, 2.34567, 3.45678]
        result = engine.calculate_basic_stats(data)

        assert result.mean == 2.34  # Rounded to 2 decimal places

    # Correlation Tests

    def test_calculate_correlation_perfect_positive(self):
        """Test perfect positive correlation."""
        x_data = [1, 2, 3, 4, 5]
        y_data = [2, 4, 6, 8, 10]
        correlation = self.engine.calculate_correlation(x_data, y_data)

        assert correlation is not None
        assert abs(correlation - 1.0) < 0.0001

    def test_calculate_correlation_perfect_negative(self):
        """Test perfect negative correlation."""
        x_data = [1, 2, 3, 4, 5]
        y_data = [10, 8, 6, 4, 2]
        correlation = self.engine.calculate_correlation(x_data, y_data)

        assert correlation is not None
        assert abs(correlation + 1.0) < 0.0001

    def test_calculate_correlation_no_correlation(self):
        """Test no correlation (should be near zero)."""
        x_data = [1, 2, 3, 4, 5]
        y_data = [3, 1, 4, 2, 5]  # Random order
        correlation = self.engine.calculate_correlation(x_data, y_data)

        assert correlation is not None
        assert abs(correlation) < 0.8  # Should be relatively low

    def test_calculate_correlation_different_lengths(self):
        """Test correlation with different length datasets."""
        x_data = [1, 2, 3]
        y_data = [1, 2, 3, 4, 5]
        correlation = self.engine.calculate_correlation(x_data, y_data)

        assert correlation is None

    def test_calculate_correlation_insufficient_data(self):
        """Test correlation with insufficient data."""
        x_data = [1]
        y_data = [2]
        correlation = self.engine.calculate_correlation(x_data, y_data)

        assert correlation is None

    def test_calculate_correlation_with_nan_values(self):
        """Test correlation handling NaN values."""
        x_data = [1, 2, float('nan'), 4, 5]
        y_data = [2, 4, 6, 8, 10]
        correlation = self.engine.calculate_correlation(x_data, y_data)

        assert correlation is not None
        assert abs(correlation - 1.0) < 0.0001  # Should still be perfect correlation

    def test_calculate_correlation_with_infinite_values(self):
        """Test correlation handling infinite values."""
        x_data = [1, 2, float('inf'), 4, 5]
        y_data = [2, 4, 6, 8, 10]
        correlation = self.engine.calculate_correlation(x_data, y_data)

        assert correlation is not None
        assert abs(correlation - 1.0) < 0.0001

    def test_calculate_correlation_constant_values(self):
        """Test correlation with constant values (zero variance)."""
        x_data = [5, 5, 5, 5, 5]
        y_data = [1, 2, 3, 4, 5]
        correlation = self.engine.calculate_correlation(x_data, y_data)

        assert correlation is None  # Cannot calculate with zero variance

    def test_calculate_correlation_empty_after_filtering(self):
        """Test correlation when all values are invalid."""
        x_data = [float('nan'), float('inf')]
        y_data = [float('nan'), float('-inf')]
        correlation = self.engine.calculate_correlation(x_data, y_data)

        assert correlation is None

    # Moving Average Tests

    def test_calculate_moving_average_normal_data(self):
        """Test moving average with normal data."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        window_size = 3
        result = self.engine.calculate_moving_average(data, window_size)

        expected = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        assert len(result) == len(data)
        for i, expected_val in enumerate(expected):
            assert abs(result[i] - expected_val) < 0.0001

    def test_calculate_moving_average_window_size_one(self):
        """Test moving average with window size of 1."""
        data = [1, 2, 3, 4, 5]
        result = self.engine.calculate_moving_average(data, 1)

        assert result == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_calculate_moving_average_window_larger_than_data(self):
        """Test moving average with window larger than data."""
        data = [1, 2, 3]
        result = self.engine.calculate_moving_average(data, 5)

        # All values should be average of available data up to that point
        assert result[0] == 1.0
        assert result[1] == 1.5
        assert abs(result[2] - 2.0) < 0.0001

    def test_calculate_moving_average_with_nan_values(self):
        """Test moving average handling NaN values."""
        data = [1, float('nan'), 3, 4, 5]
        result = self.engine.calculate_moving_average(data, 2)

        assert result[0] == 1.0
        assert result[1] == 1.0  # Only valid value in window
        assert result[2] == 3.0  # NaN excluded, only 3 in window
        assert result[3] == 3.5  # Average of 3 and 4

    def test_calculate_moving_average_zero_window_size(self):
        """Test moving average with zero window size."""
        data = [1, 2, 3, 4, 5]
        with pytest.raises(ValueError, match="Window size must be positive"):
            self.engine.calculate_moving_average(data, 0)

    def test_calculate_moving_average_negative_window_size(self):
        """Test moving average with negative window size."""
        data = [1, 2, 3, 4, 5]
        with pytest.raises(ValueError, match="Window size must be positive"):
            self.engine.calculate_moving_average(data, -1)

    def test_calculate_moving_average_empty_data(self):
        """Test moving average with empty data."""
        result = self.engine.calculate_moving_average([], 3)
        assert result == []

    def test_calculate_moving_average_all_invalid_values(self):
        """Test moving average with all invalid values."""
        data = [float('nan'), float('inf'), float('-inf')]
        result = self.engine.calculate_moving_average(data, 2)

        assert len(result) == 3
        assert all(val is None for val in result)

    # Outlier Detection Tests

    def test_detect_outliers_iqr_method(self):
        """Test outlier detection using IQR method."""
        # Data with clear outliers
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]  # 100 is outlier
        result = self.engine.detect_outliers(data, method="iqr", threshold=1.5)

        assert result["method"] == "iqr"
        assert result["threshold"] == 1.5
        assert len(result["outliers"]) > 0
        assert 9 in result["outliers"]  # Index of outlier value 100
        assert 100.0 in result["outlier_values"]
        assert result["total_outliers"] > 0
        assert result["outlier_percentage"] > 0

    def test_detect_outliers_zscore_method(self):
        """Test outlier detection using Z-score method."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
        result = self.engine.detect_outliers(data, method="zscore", threshold=2.0)

        assert result["method"] == "zscore"
        assert len(result["outliers"]) > 0
        assert 100.0 in result["outlier_values"]

    def test_detect_outliers_modified_zscore_method(self):
        """Test outlier detection using modified Z-score method."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
        result = self.engine.detect_outliers(data, method="modified_zscore", threshold=3.5)

        assert result["method"] == "modified_zscore"
        assert len(result["outliers"]) > 0

    def test_detect_outliers_no_outliers(self):
        """Test outlier detection with no outliers."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Normal distribution
        result = self.engine.detect_outliers(data, method="iqr", threshold=1.5)

        assert len(result["outliers"]) == 0
        assert len(result["outlier_values"]) == 0
        assert result["total_outliers"] == 0
        assert result["outlier_percentage"] == 0

    def test_detect_outliers_empty_data(self):
        """Test outlier detection with empty data."""
        result = self.engine.detect_outliers([], method="iqr")

        assert result["outliers"] == []
        assert result["outlier_values"] == []
        assert result["method"] == "iqr"

    def test_detect_outliers_insufficient_data(self):
        """Test outlier detection with insufficient data."""
        data = [1, 2]
        result = self.engine.detect_outliers(data, method="iqr")

        assert result["outliers"] == []
        assert "warning" in result
        assert "Insufficient valid data" in result["warning"]

    def test_detect_outliers_with_nan_values(self):
        """Test outlier detection handling NaN values."""
        data = [1, 2, 3, float('nan'), 5, 6, 7, 8, 100]
        result = self.engine.detect_outliers(data, method="iqr", threshold=1.5)

        # Should still detect outlier (100) while ignoring NaN
        assert len(result["outliers"]) > 0
        assert 100.0 in result["outlier_values"]

    def test_detect_outliers_constant_values_zscore(self):
        """Test outlier detection with constant values (zero std dev)."""
        data = [5, 5, 5, 5, 5]
        result = self.engine.detect_outliers(data, method="zscore", threshold=2.0)

        assert len(result["outliers"]) == 0  # No outliers with zero variation

    def test_detect_outliers_constant_values_modified_zscore(self):
        """Test outlier detection with constant values (zero MAD)."""
        data = [5, 5, 5, 5, 5]
        result = self.engine.detect_outliers(data, method="modified_zscore", threshold=3.5)

        assert len(result["outliers"]) == 0

    def test_detect_outliers_unknown_method(self):
        """Test outlier detection with unknown method."""
        data = [1, 2, 3, 4, 5]
        result = self.engine.detect_outliers(data, method="unknown", threshold=1.5)

        assert result["outliers"] == []
        assert result["outlier_values"] == []
        assert "error" in result
        assert "Unknown outlier detection method" in result["error"]

    # Percentile Tests

    def test_percentile_calculations(self):
        """Test percentile calculations in basic stats."""
        data = list(range(1, 101))  # 1 to 100
        result = self.engine.calculate_basic_stats(data)

        assert result.percentiles is not None
        assert 50 in result.percentiles
        assert abs(result.percentiles[50] - 50.5) < 0.1  # Median should be around 50.5
        assert 25 in result.percentiles
        assert 75 in result.percentiles

    def test_percentiles_small_dataset(self):
        """Test percentile calculations with small dataset."""
        data = [1, 2, 3]
        result = self.engine.calculate_basic_stats(data)

        assert result.percentiles is not None
        assert 50 in result.percentiles
        assert result.percentiles[50] == 2.0  # Median

    def test_percentiles_single_value(self):
        """Test percentile calculations with single value."""
        data = [42]
        result = self.engine.calculate_basic_stats(data)

        assert result.percentiles is not None
        for percentile in [25, 50, 75]:
            if percentile in result.percentiles:
                assert result.percentiles[percentile] == 42.0

    # Caching Tests

    def test_caching_functionality(self):
        """Test result caching."""
        data = [1, 2, 3, 4, 5]
        cache_key = "test_cache"

        # First call should calculate and cache
        result1 = self.engine.calculate_basic_stats(data, cache_key=cache_key)

        # Second call should use cache
        result2 = self.engine.calculate_basic_stats(data, cache_key=cache_key)

        assert result1.mean == result2.mean
        assert cache_key in self.engine._stats_cache

    def test_cache_without_key(self):
        """Test that results are not cached without cache key."""
        data = [1, 2, 3, 4, 5]
        result = self.engine.calculate_basic_stats(data)

        assert len(self.engine._stats_cache) == 0

    @patch('src.python.workspace_qdrant_mcp.analytics.engine.statistical_engine.datetime')
    def test_cache_expiry(self, mock_datetime):
        """Test cache expiry functionality."""
        # Mock datetime to control cache timing
        start_time = datetime(2023, 1, 1, 12, 0, 0)
        expired_time = start_time + timedelta(minutes=10)  # Beyond TTL

        mock_datetime.now.side_effect = [start_time, start_time, expired_time]

        data = [1, 2, 3, 4, 5]
        cache_key = "test_expiry"

        # First call - cache result
        self.engine.calculate_basic_stats(data, cache_key=cache_key)

        # Check cache is valid initially
        assert self.engine._is_cached_valid(cache_key) is False  # Expired time

    def test_clear_cache(self):
        """Test cache clearing."""
        data = [1, 2, 3, 4, 5]
        cache_key = "test_clear"

        # Cache a result
        self.engine.calculate_basic_stats(data, cache_key=cache_key)
        assert len(self.engine._stats_cache) > 0

        # Clear cache
        self.engine.clear_cache()
        assert len(self.engine._stats_cache) == 0
        assert len(self.engine._cache_timestamps) == 0

    # Error Handling Tests

    def test_error_handling_in_basic_stats(self):
        """Test error handling in basic statistics calculation."""
        # Mock an exception during calculation
        with patch('statistics.mean', side_effect=OverflowError("Test overflow")):
            data = [1, 2, 3, 4, 5]
            result = self.engine.calculate_basic_stats(data)

            # Should handle error gracefully
            assert result.mean is None
            assert result.is_valid is True  # Other calculations might still work

    def test_unexpected_error_handling(self):
        """Test handling of unexpected errors."""
        # Mock an unexpected exception
        with patch('statistics.median', side_effect=RuntimeError("Unexpected error")):
            data = [1, 2, 3, 4, 5]
            result = self.engine.calculate_basic_stats(data)

            # Should handle error and continue with other calculations
            assert result.median is None

    def test_correlation_error_handling(self):
        """Test error handling in correlation calculation."""
        with patch('statistics.correlation', side_effect=ValueError("Test error")):
            x_data = [1, 2, 3]
            y_data = [1, 2, 3]
            correlation = self.engine.calculate_correlation(x_data, y_data)

            assert correlation is None

    # Integration Tests

    def test_comprehensive_analysis_workflow(self):
        """Test comprehensive analysis workflow."""
        # Large dataset with various characteristics
        import random
        random.seed(42)  # For reproducible results

        data = []
        # Normal values
        for i in range(100):
            data.append(random.gauss(50, 10))

        # Add some outliers
        data.extend([150, -50, 200])

        # Add some invalid values
        data.extend([float('nan'), float('inf')])

        # Run comprehensive analysis
        stats_result = self.engine.calculate_basic_stats(data)
        outliers_result = self.engine.detect_outliers(data, method="iqr")
        moving_avg = self.engine.calculate_moving_average(data[:10], 3)

        # Verify all analyses completed successfully
        assert stats_result.is_valid is True
        assert stats_result.count > 0
        assert len(outliers_result["outliers"]) > 0
        assert len(moving_avg) == 10

    def test_large_dataset_performance(self):
        """Test performance with large dataset."""
        # Create large dataset
        data = list(range(10000))

        # Should complete without errors
        result = self.engine.calculate_basic_stats(data)
        assert result.is_valid is True
        assert result.count == 10000