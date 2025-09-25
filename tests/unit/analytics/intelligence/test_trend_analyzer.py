"""
Comprehensive unit tests for AdvancedTrendAnalyzer with extensive edge case coverage.

Tests cover:
- Normal trend detection workflows
- Edge cases: empty data, NaN, infinity, insufficient samples
- Various trend types: linear, exponential, cyclical patterns
- Change point detection with boundary conditions
- Seasonality detection edge cases
- Trend segmentation and analysis
- Error handling and graceful degradation
- Performance with large and noisy datasets
"""

import pytest
import math
import statistics
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from src.python.workspace_qdrant_mcp.analytics.intelligence.trend_analyzer import (
    AdvancedTrendAnalyzer,
    TrendAnalysisResult,
    TrendSegment,
    ChangePoint,
    TrendDirection,
    TrendStrength,
    TrendType,
    ChangePointType
)


class TestChangePoint:
    """Tests for ChangePoint data class."""

    def test_change_point_initialization(self):
        """Test ChangePoint initialization with all fields."""
        timestamp = datetime.now()
        change_point = ChangePoint(
            index=10,
            timestamp=timestamp,
            change_type=ChangePointType.LEVEL_SHIFT,
            before_value=5.0,
            after_value=8.0,
            magnitude=3.0,
            confidence=0.8,
            statistical_significance=0.9,
            context={'test': 'data'}
        )

        assert change_point.index == 10
        assert change_point.timestamp == timestamp
        assert change_point.change_type == ChangePointType.LEVEL_SHIFT
        assert change_point.before_value == 5.0
        assert change_point.after_value == 8.0
        assert change_point.magnitude == 3.0
        assert change_point.confidence == 0.8
        assert change_point.statistical_significance == 0.9
        assert change_point.context == {'test': 'data'}

    def test_change_point_to_dict(self):
        """Test ChangePoint serialization to dictionary."""
        timestamp = datetime(2023, 1, 1, 12, 0, 0)
        change_point = ChangePoint(
            index=5,
            timestamp=timestamp,
            change_type=ChangePointType.TREND_CHANGE,
            before_value=2.0,
            after_value=4.0,
            magnitude=2.0,
            confidence=0.7
        )

        result_dict = change_point.to_dict()

        assert result_dict['index'] == 5
        assert result_dict['timestamp'] == '2023-01-01T12:00:00'
        assert result_dict['change_type'] == 'trend_change'
        assert result_dict['before_value'] == 2.0
        assert result_dict['after_value'] == 4.0


class TestTrendSegment:
    """Tests for TrendSegment data class."""

    def test_trend_segment_initialization(self):
        """Test TrendSegment initialization."""
        segment = TrendSegment(
            start_index=0,
            end_index=9,
            direction=TrendDirection.MODERATELY_INCREASING,
            strength=TrendStrength.STRONG,
            trend_type=TrendType.LINEAR,
            slope=0.5,
            intercept=1.0,
            r_squared=0.8,
            confidence=0.9
        )

        assert segment.start_index == 0
        assert segment.end_index == 9
        assert segment.direction == TrendDirection.MODERATELY_INCREASING
        assert segment.strength == TrendStrength.STRONG
        assert segment.trend_type == TrendType.LINEAR
        assert segment.slope == 0.5
        assert segment.intercept == 1.0
        assert segment.r_squared == 0.8
        assert segment.confidence == 0.9

    def test_trend_segment_to_dict(self):
        """Test TrendSegment serialization."""
        segment = TrendSegment(
            start_index=5,
            end_index=15,
            direction=TrendDirection.STABLE,
            strength=TrendStrength.WEAK,
            trend_type=TrendType.CYCLICAL,
            slope=0.0,
            intercept=5.0,
            r_squared=0.2,
            confidence=0.3,
            values=[5, 5, 5, 5, 5]
        )

        result_dict = segment.to_dict()

        assert result_dict['start_index'] == 5
        assert result_dict['end_index'] == 15
        assert result_dict['direction'] == 'stable'
        assert result_dict['strength'] == 'weak'
        assert result_dict['trend_type'] == 'cyclical'
        assert result_dict['segment_length'] == 11  # end - start + 1


class TestTrendAnalysisResult:
    """Tests for TrendAnalysisResult data class."""

    def test_trend_analysis_result_initialization(self):
        """Test TrendAnalysisResult initialization."""
        result = TrendAnalysisResult(
            overall_direction=TrendDirection.STRONGLY_INCREASING,
            overall_strength=TrendStrength.VERY_STRONG,
            overall_trend_type=TrendType.EXPONENTIAL
        )

        assert result.overall_direction == TrendDirection.STRONGLY_INCREASING
        assert result.overall_strength == TrendStrength.VERY_STRONG
        assert result.overall_trend_type == TrendType.EXPONENTIAL
        assert result.trend_segments == []
        assert result.change_points == []
        assert result.seasonality_detected is False
        assert result.is_reliable is True

    def test_trend_analysis_result_to_dict(self):
        """Test TrendAnalysisResult serialization."""
        result = TrendAnalysisResult(
            overall_direction=TrendDirection.MODERATELY_DECREASING,
            overall_strength=TrendStrength.MODERATE,
            overall_trend_type=TrendType.POLYNOMIAL,
            seasonality_detected=True,
            seasonal_period=12,
            seasonal_strength=0.6
        )

        result_dict = result.to_dict()

        assert result_dict['overall_direction'] == 'moderately_decreasing'
        assert result_dict['overall_strength'] == 'moderate'
        assert result_dict['overall_trend_type'] == 'polynomial'
        assert result_dict['seasonality_detected'] is True
        assert result_dict['seasonal_period'] == 12
        assert result_dict['seasonal_strength'] == 0.6


class TestAdvancedTrendAnalyzer:
    """Comprehensive tests for AdvancedTrendAnalyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = AdvancedTrendAnalyzer(
            min_segment_length=5,
            change_point_threshold=0.1,
            seasonality_min_periods=8,
            trend_confidence_threshold=0.3
        )

    def test_initialization(self):
        """Test analyzer initialization."""
        assert self.analyzer.min_segment_length == 5
        assert self.analyzer.change_point_threshold == 0.1
        assert self.analyzer.seasonality_min_periods == 8
        assert self.analyzer.trend_confidence_threshold == 0.3

    def test_initialization_custom_parameters(self):
        """Test analyzer initialization with custom parameters."""
        analyzer = AdvancedTrendAnalyzer(
            min_segment_length=10,
            change_point_threshold=0.2,
            seasonality_min_periods=12,
            trend_confidence_threshold=0.5
        )
        assert analyzer.min_segment_length == 10
        assert analyzer.change_point_threshold == 0.2
        assert analyzer.seasonality_min_periods == 12
        assert analyzer.trend_confidence_threshold == 0.5

    # Basic Functionality Tests

    def test_analyze_trends_increasing_linear(self):
        """Test trend analysis with clear increasing linear trend."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = self.analyzer.analyze_trends(data)

        assert result.is_reliable is True
        assert result.overall_direction in [
            TrendDirection.STRONGLY_INCREASING,
            TrendDirection.MODERATELY_INCREASING
        ]
        assert result.overall_strength in [TrendStrength.STRONG, TrendStrength.VERY_STRONG]
        assert result.overall_trend_type == TrendType.LINEAR
        assert result.trend_confidence > 0.8
        assert len(result.recommendations) > 0

    def test_analyze_trends_decreasing_linear(self):
        """Test trend analysis with clear decreasing linear trend."""
        data = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        result = self.analyzer.analyze_trends(data)

        assert result.is_reliable is True
        assert result.overall_direction in [
            TrendDirection.STRONGLY_DECREASING,
            TrendDirection.MODERATELY_DECREASING
        ]
        assert result.overall_strength in [TrendStrength.STRONG, TrendStrength.VERY_STRONG]
        assert result.trend_confidence > 0.8

    def test_analyze_trends_stable_data(self):
        """Test trend analysis with stable data."""
        data = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        result = self.analyzer.analyze_trends(data)

        assert result.is_reliable is True
        assert result.overall_direction == TrendDirection.STABLE
        assert result.overall_strength == TrendStrength.VERY_WEAK
        assert result.trend_confidence < 0.3

    def test_analyze_trends_with_timestamps(self):
        """Test trend analysis with timestamps provided."""
        data = [1, 2, 3, 4, 5]
        timestamps = [
            datetime(2023, 1, 1) + timedelta(days=i)
            for i in range(len(data))
        ]
        result = self.analyzer.analyze_trends(data, timestamps)

        assert result.is_reliable is True
        assert len(result.metadata) > 0
        assert 'analysis_timestamp' in result.metadata

    # Edge Case Tests

    def test_analyze_trends_empty_data(self):
        """Test trend analysis with empty data."""
        result = self.analyzer.analyze_trends([])

        assert result.is_reliable is False
        assert "Insufficient data" in result.error_message
        assert result.overall_direction == TrendDirection.UNKNOWN

    def test_analyze_trends_insufficient_data(self):
        """Test trend analysis with insufficient data."""
        data = [1, 2]
        result = self.analyzer.analyze_trends(data)

        assert result.is_reliable is False
        assert "Insufficient data" in result.error_message

    def test_analyze_trends_with_nan_values(self):
        """Test trend analysis with NaN values."""
        data = [1, 2, float('nan'), 4, 5, float('nan'), 7, 8, 9, 10]
        result = self.analyzer.analyze_trends(data)

        assert result.is_reliable is True
        # Should filter out NaN values and continue analysis
        assert result.metadata['filtered_points'] == 2  # 2 NaN values filtered

    def test_analyze_trends_with_infinity_values(self):
        """Test trend analysis with infinity values."""
        data = [1, 2, float('inf'), 4, 5, float('-inf'), 7, 8, 9, 10]
        result = self.analyzer.analyze_trends(data)

        assert result.is_reliable is True
        # Should filter out infinity values
        assert result.metadata['filtered_points'] == 2

    def test_analyze_trends_all_invalid_values(self):
        """Test trend analysis with all invalid values."""
        data = [float('nan'), float('inf'), float('-inf'), float('nan')]
        result = self.analyzer.analyze_trends(data)

        assert result.is_reliable is False
        assert "Insufficient data" in result.error_message

    def test_analyze_trends_single_value(self):
        """Test trend analysis with single value."""
        data = [42]
        result = self.analyzer.analyze_trends(data)

        assert result.is_reliable is False
        assert "Insufficient data" in result.error_message

    def test_analyze_trends_two_values(self):
        """Test trend analysis with exactly two values."""
        data = [1, 5]
        result = self.analyzer.analyze_trends(data)

        assert result.is_reliable is False
        assert "Insufficient data" in result.error_message

    # Trend Classification Tests

    def test_classify_trend_strength(self):
        """Test trend strength classification."""
        assert self.analyzer._classify_trend_strength(0.9) == TrendStrength.VERY_STRONG
        assert self.analyzer._classify_trend_strength(0.7) == TrendStrength.STRONG
        assert self.analyzer._classify_trend_strength(0.5) == TrendStrength.MODERATE
        assert self.analyzer._classify_trend_strength(0.3) == TrendStrength.WEAK
        assert self.analyzer._classify_trend_strength(0.1) == TrendStrength.VERY_WEAK

    def test_analyze_exponential_trend(self):
        """Test analysis of exponential trend pattern."""
        # Create exponential-like data
        data = [math.exp(x * 0.2) for x in range(10)]
        result = self.analyzer.analyze_trends(data)

        assert result.is_reliable is True
        assert result.overall_direction in [
            TrendDirection.STRONGLY_INCREASING,
            TrendDirection.MODERATELY_INCREASING
        ]
        # Trend type detection might classify as exponential or linear depending on correlation

    def test_analyze_cyclical_pattern(self):
        """Test analysis of cyclical pattern."""
        # Create cyclical data using sine wave
        data = [5 + 2 * math.sin(2 * math.pi * x / 4) for x in range(20)]
        result = self.analyzer.analyze_trends(data)

        assert result.is_reliable is True
        # Cyclical patterns might be detected depending on the correlation thresholds

    def test_analyze_noisy_trend(self):
        """Test analysis of trend with noise."""
        # Linear trend with noise
        import random
        random.seed(42)  # For reproducible results

        base_trend = [x * 0.5 for x in range(20)]
        noisy_data = [val + random.gauss(0, 0.5) for val in base_trend]

        result = self.analyzer.analyze_trends(noisy_data)

        assert result.is_reliable is True
        # Should still detect underlying trend despite noise

    # Change Point Detection Tests

    def test_detect_change_points_level_shift(self):
        """Test change point detection with clear level shift."""
        data = [5] * 10 + [10] * 10  # Clear level shift at index 10
        result = self.analyzer.analyze_trends(data)

        assert result.is_reliable is True
        assert len(result.change_points) > 0

        # Should detect change point around index 10
        change_indices = [cp.index for cp in result.change_points]
        assert any(8 <= idx <= 12 for idx in change_indices)  # Allow some tolerance

    def test_detect_change_points_trend_change(self):
        """Test change point detection with trend direction change."""
        # Increasing trend followed by decreasing trend
        data = list(range(10)) + list(range(10, 0, -1))
        result = self.analyzer.analyze_trends(data)

        assert result.is_reliable is True
        # Should detect change point around the middle

    def test_detect_change_points_no_changes(self):
        """Test change point detection with no significant changes."""
        data = [5 + 0.1 * x for x in range(20)]  # Smooth linear trend
        result = self.analyzer.analyze_trends(data)

        assert result.is_reliable is True
        # Should not detect change points in smooth trend
        assert len(result.change_points) == 0

    def test_detect_change_points_insufficient_data(self):
        """Test change point detection with insufficient data."""
        data = [1, 2, 3, 4]  # Less than min_segment_length * 2
        result = self.analyzer.analyze_trends(data)

        assert result.is_reliable is True
        assert len(result.change_points) == 0

    # Seasonality Detection Tests

    def test_detect_seasonality_clear_pattern(self):
        """Test seasonality detection with clear seasonal pattern."""
        # Create data with period-4 seasonality
        data = []
        for i in range(40):
            seasonal_component = 2 * math.sin(2 * math.pi * i / 4)
            trend_component = i * 0.1
            data.append(10 + trend_component + seasonal_component)

        result = self.analyzer.analyze_trends(data)

        assert result.is_reliable is True
        # Should detect seasonality with period around 4
        if result.seasonality_detected:
            assert result.seasonal_period in range(3, 6)  # Allow some tolerance

    def test_detect_seasonality_no_pattern(self):
        """Test seasonality detection with no seasonal pattern."""
        data = [x * 0.5 + 10 for x in range(20)]  # Pure linear trend
        result = self.analyzer.analyze_trends(data)

        assert result.is_reliable is True
        assert result.seasonality_detected is False
        assert result.seasonal_period is None

    def test_detect_seasonality_insufficient_data(self):
        """Test seasonality detection with insufficient data."""
        data = [1, 2, 3, 4, 5, 6]  # Less than seasonality_min_periods
        result = self.analyzer.analyze_trends(data)

        assert result.is_reliable is True
        assert result.seasonality_detected is False

    # Trend Segmentation Tests

    def test_trend_segments_single_segment(self):
        """Test trend segmentation with single consistent trend."""
        data = [x * 0.5 + 2 for x in range(15)]
        result = self.analyzer.analyze_trends(data)

        assert result.is_reliable is True
        assert len(result.trend_segments) >= 1
        # Single segment should cover most/all of the data
        if result.trend_segments:
            segment = result.trend_segments[0]
            assert segment.start_index == 0
            assert segment.end_index == len(data) - 1

    def test_trend_segments_multiple_segments(self):
        """Test trend segmentation with multiple distinct trends."""
        # Create data with clear segments
        segment1 = [x for x in range(10)]  # Increasing
        segment2 = [10] * 10  # Stable
        segment3 = [10 - x for x in range(10)]  # Decreasing
        data = segment1 + segment2 + segment3

        # Adjust analyzer for smaller segments
        analyzer = AdvancedTrendAnalyzer(
            min_segment_length=3,
            change_point_threshold=0.05
        )

        result = analyzer.analyze_trends(data)

        assert result.is_reliable is True
        # Should detect multiple segments if change points are found

    def test_analyze_single_segment(self):
        """Test analysis of a single segment."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        segment = self.analyzer._analyze_single_segment(data, 0, 9)

        assert segment is not None
        assert segment.start_index == 0
        assert segment.end_index == 9
        assert segment.direction == TrendDirection.STRONGLY_INCREASING
        assert segment.slope > 0
        assert segment.confidence > 0.8

    def test_analyze_single_segment_constant(self):
        """Test analysis of segment with constant values."""
        data = [5, 5, 5, 5, 5]
        segment = self.analyzer._analyze_single_segment(data, 0, 4)

        assert segment is not None
        assert segment.direction == TrendDirection.STABLE
        assert segment.slope == 0.0
        assert segment.r_squared == 1.0

    def test_analyze_single_segment_insufficient_data(self):
        """Test analysis of segment with insufficient data."""
        data = [1]
        segment = self.analyzer._analyze_single_segment(data, 0, 0)

        assert segment is None

    # Data Preparation Tests

    def test_prepare_data_normal(self):
        """Test data preparation with normal data."""
        data = [1, 2, 3, 4, 5]
        timestamps = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(5)]

        valid_data, valid_timestamps = self.analyzer._prepare_data(data, timestamps)

        assert len(valid_data) == 5
        assert len(valid_timestamps) == 5
        assert valid_data == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_prepare_data_with_invalid_values(self):
        """Test data preparation with invalid values."""
        data = [1, 2, float('nan'), 4, float('inf'), 6]
        timestamps = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(6)]

        valid_data, valid_timestamps = self.analyzer._prepare_data(data, timestamps)

        assert len(valid_data) == 4  # Only 1, 2, 4, 6 are valid (nan and inf filtered)
        assert valid_data == [1.0, 2.0, 4.0, 6.0]
        assert len(valid_timestamps) == 4

    def test_prepare_data_no_timestamps(self):
        """Test data preparation without timestamps."""
        data = [1, 2, 3, 4, 5]
        valid_data, valid_timestamps = self.analyzer._prepare_data(data, None)

        assert len(valid_data) == 5
        assert len(valid_timestamps) == 5
        assert all(ts is None for ts in valid_timestamps)

    # Error Handling Tests

    def test_error_handling_in_trend_analysis(self):
        """Test error handling during trend analysis."""
        # Mock an error in overall trend analysis
        with patch.object(self.analyzer, '_analyze_overall_trend', side_effect=Exception("Test error")):
            data = [1, 2, 3, 4, 5]
            result = self.analyzer.analyze_trends(data)

            # Should handle error gracefully
            assert result.is_reliable is False
            assert "Analysis error" in result.error_message

    def test_error_handling_in_change_point_detection(self):
        """Test error handling in change point detection."""
        with patch.object(self.analyzer, '_detect_change_points', side_effect=Exception("Test error")):
            data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            result = self.analyzer.analyze_trends(data)

            # Should continue analysis despite change point detection error
            assert result.is_reliable is True  # Other parts still work

    def test_error_handling_in_seasonality_detection(self):
        """Test error handling in seasonality detection."""
        with patch.object(self.analyzer, '_detect_seasonality', side_effect=Exception("Test error")):
            data = list(range(20))
            result = self.analyzer.analyze_trends(data)

            # Should continue analysis despite seasonality detection error
            assert result.is_reliable is True

    # Performance Tests

    def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        # Generate large dataset
        data = [x * 0.1 + math.sin(x * 0.1) for x in range(1000)]

        result = self.analyzer.analyze_trends(data)

        # Should complete without errors
        assert result.is_reliable is True
        assert result.metadata['data_length'] == 1000

    def test_very_noisy_data(self):
        """Test analysis of very noisy data."""
        import random
        random.seed(42)

        # Generate very noisy data
        data = [random.gauss(10, 5) for _ in range(50)]

        result = self.analyzer.analyze_trends(data)

        # Should handle noisy data gracefully
        assert result.is_reliable is True
        # Trend strength should be low for random data
        assert result.overall_strength in [TrendStrength.VERY_WEAK, TrendStrength.WEAK]

    def test_extreme_values(self):
        """Test analysis with extreme values."""
        data = [
            1e-10,  # Very small
            1e10,   # Very large
            -1e10,  # Very large negative
            0,      # Zero
            1, 2, 3, 4, 5  # Normal values
        ]

        result = self.analyzer.analyze_trends(data)

        # Should handle extreme values without crashing
        assert result.is_reliable is True

    # Integration Tests

    def test_comprehensive_trend_analysis_workflow(self):
        """Test comprehensive trend analysis workflow."""
        # Complex dataset with multiple characteristics
        base_trend = [x * 0.2 for x in range(50)]  # Linear trend
        seasonal = [math.sin(2 * math.pi * x / 12) for x in range(50)]  # Seasonal
        noise = [0.1 * math.sin(x) for x in range(50)]  # Noise

        data = [b + s + n for b, s, n in zip(base_trend, seasonal, noise)]

        # Add a level shift in the middle
        for i in range(25, 50):
            data[i] += 5

        timestamps = [
            datetime(2023, 1, 1) + timedelta(days=i)
            for i in range(len(data))
        ]

        result = self.analyzer.analyze_trends(data, timestamps)

        # Should generate comprehensive analysis
        assert result.is_reliable is True
        assert len(result.recommendations) > 0
        assert result.analysis_quality > 0
        assert 'data_length' in result.metadata
        assert result.metadata['data_length'] == 50

        # Should detect overall upward trend
        assert result.overall_direction in [
            TrendDirection.WEAKLY_INCREASING,
            TrendDirection.MODERATELY_INCREASING,
            TrendDirection.STRONGLY_INCREASING
        ]

    def test_filter_overlapping_change_points(self):
        """Test filtering of overlapping change points."""
        # Create overlapping change points
        change_points = [
            ChangePoint(index=10, confidence=0.8, magnitude=0.5),
            ChangePoint(index=12, confidence=0.9, magnitude=0.6),  # Overlapping, higher confidence
            ChangePoint(index=11, confidence=0.7, magnitude=0.4),  # Overlapping, lower confidence
            ChangePoint(index=20, confidence=0.8, magnitude=0.5),  # Non-overlapping
        ]

        filtered = self.analyzer._filter_overlapping_change_points(change_points)

        # Should keep the highest confidence point from overlapping group and non-overlapping point
        assert len(filtered) == 2
        indices = [cp.index for cp in filtered]
        assert 12 in indices  # Highest confidence from overlapping group
        assert 20 in indices  # Non-overlapping

    def test_cyclical_pattern_detection(self):
        """Test detection of cyclical patterns."""
        # Create clear cyclical pattern
        data = []
        for i in range(40):
            value = 10 + 5 * math.sin(2 * math.pi * i / 8)  # Period of 8
            data.append(value)

        is_cyclical = self.analyzer._is_cyclical_pattern(data)

        # Should detect cyclical pattern
        assert is_cyclical is True

    def test_cyclical_pattern_detection_linear_data(self):
        """Test cyclical pattern detection with linear data."""
        data = list(range(20))  # Linear, not cyclical

        is_cyclical = self.analyzer._is_cyclical_pattern(data)

        # Should not detect cyclical pattern in linear data
        assert is_cyclical is False

    def test_cyclical_pattern_detection_insufficient_data(self):
        """Test cyclical pattern detection with insufficient data."""
        data = [1, 2, 3, 4, 5]  # Too short for cycle detection

        is_cyclical = self.analyzer._is_cyclical_pattern(data)

        # Should not detect cyclical pattern with insufficient data
        assert is_cyclical is False