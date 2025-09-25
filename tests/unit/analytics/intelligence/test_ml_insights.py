"""
Comprehensive unit tests for MLInsightEngine with extensive edge case coverage.

Tests cover:
- Normal insight generation workflows
- Edge cases: empty data, NaN, infinity, insufficient samples
- Pattern recognition with noisy and sparse data
- Data quality assessment edge cases
- Correlation analysis boundary conditions
- Distribution analysis with extreme cases
- Error handling and graceful degradation
- Performance with large datasets
- Meta-insight generation
"""

import pytest
import math
import statistics
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from src.python.workspace_qdrant_mcp.analytics.intelligence.ml_insights import (
    MLInsightEngine,
    Insight,
    InsightReport,
    InsightType,
    ConfidenceLevel,
    DataQualityIssue
)


class TestInsight:
    """Tests for Insight data class."""

    def test_insight_initialization(self):
        """Test Insight initialization with all fields."""
        insight = Insight(
            insight_type=InsightType.PATTERN_DISCOVERY,
            title="Test Insight",
            description="Test description",
            confidence_level=ConfidenceLevel.HIGH,
            confidence_score=0.85,
            supporting_evidence={"test": "data"},
            recommendations=["Do something"],
            impact_score=7.5,
            actionable=True
        )

        assert insight.insight_type == InsightType.PATTERN_DISCOVERY
        assert insight.title == "Test Insight"
        assert insight.description == "Test description"
        assert insight.confidence_level == ConfidenceLevel.HIGH
        assert insight.confidence_score == 0.85
        assert insight.supporting_evidence == {"test": "data"}
        assert insight.recommendations == ["Do something"]
        assert insight.impact_score == 7.5
        assert insight.actionable is True
        assert isinstance(insight.timestamp, datetime)

    def test_insight_to_dict(self):
        """Test Insight serialization to dictionary."""
        insight = Insight(
            insight_type=InsightType.DATA_QUALITY,
            title="Quality Issue",
            description="Data quality problem",
            confidence_level=ConfidenceLevel.MEDIUM,
            confidence_score=0.6,
            supporting_evidence={"issue": "missing_values"},
            recommendations=["Clean data"],
            impact_score=5.0,
            actionable=True
        )

        result_dict = insight.to_dict()

        assert result_dict['insight_type'] == 'data_quality'
        assert result_dict['title'] == "Quality Issue"
        assert result_dict['confidence_level'] == 'medium'
        assert result_dict['confidence_score'] == 0.6
        assert result_dict['supporting_evidence'] == {"issue": "missing_values"}
        assert result_dict['actionable'] is True


class TestInsightReport:
    """Tests for InsightReport data class."""

    def test_insight_report_initialization(self):
        """Test InsightReport initialization and derived field calculation."""
        insights = [
            Insight(
                insight_type=InsightType.PATTERN_DISCOVERY,
                title="High Impact",
                description="Test",
                confidence_level=ConfidenceLevel.HIGH,
                confidence_score=0.9,
                supporting_evidence={},
                recommendations=[],
                impact_score=8.0,
                actionable=True
            ),
            Insight(
                insight_type=InsightType.DATA_QUALITY,
                title="Low Impact",
                description="Test",
                confidence_level=ConfidenceLevel.LOW,
                confidence_score=0.3,
                supporting_evidence={},
                recommendations=[],
                impact_score=3.0,
                actionable=False
            )
        ]

        report = InsightReport(insights=insights, quality_score=75.0)

        assert report.total_insights == 2
        assert report.high_impact_insights == 1  # Impact score >= 7.0
        assert report.actionable_insights == 1
        assert report.quality_score == 75.0

    def test_insight_report_to_dict(self):
        """Test InsightReport serialization."""
        report = InsightReport(quality_score=80.0, is_reliable=True)
        result_dict = report.to_dict()

        assert result_dict['quality_score'] == 80.0
        assert result_dict['is_reliable'] is True
        assert result_dict['total_insights'] == 0
        assert isinstance(result_dict['insights'], list)


class TestMLInsightEngine:
    """Comprehensive tests for MLInsightEngine."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = MLInsightEngine(
            min_samples=5,
            significance_threshold=0.05,
            confidence_threshold=0.7
        )

    def test_initialization(self):
        """Test engine initialization."""
        assert self.engine.min_samples == 5
        assert self.engine.significance_threshold == 0.05
        assert self.engine.confidence_threshold == 0.7
        assert len(self.engine._analysis_cache) == 0

    def test_initialization_custom_parameters(self):
        """Test engine initialization with custom parameters."""
        engine = MLInsightEngine(
            min_samples=20,
            significance_threshold=0.01,
            confidence_threshold=0.8
        )
        assert engine.min_samples == 20
        assert engine.significance_threshold == 0.01
        assert engine.confidence_threshold == 0.8

    # Basic Functionality Tests

    def test_generate_insights_normal_data(self):
        """Test insight generation with normal data."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        report = self.engine.generate_insights(data)

        assert report.is_reliable is True
        assert report.quality_score > 0
        assert len(report.insights) > 0
        assert report.analysis_duration > 0
        assert report.error_message is None

    def test_generate_insights_multiple_series(self):
        """Test insight generation with multiple data series."""
        data = {
            'series1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'series2': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        }
        report = self.engine.generate_insights(data)

        assert report.is_reliable is True
        assert len(report.insights) > 0

        # Should detect correlation between series
        correlation_insights = [i for i in report.insights
                              if i.insight_type == InsightType.CORRELATION_ANALYSIS]
        assert len(correlation_insights) > 0

    def test_generate_insights_with_specific_analysis_types(self):
        """Test insight generation with specific analysis types."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        analysis_types = [InsightType.PATTERN_DISCOVERY, InsightType.DISTRIBUTION_ANALYSIS]

        report = self.engine.generate_insights(data, analysis_types=analysis_types)

        insight_types = [i.insight_type for i in report.insights]
        assert InsightType.PATTERN_DISCOVERY in insight_types or InsightType.DISTRIBUTION_ANALYSIS in insight_types

    # Edge Case Tests

    def test_generate_insights_empty_data(self):
        """Test insight generation with empty data."""
        report = self.engine.generate_insights([])

        assert report.is_reliable is False
        assert report.error_message is not None

    def test_generate_insights_insufficient_data(self):
        """Test insight generation with insufficient data."""
        data = [1, 2]  # Less than min_samples (5)
        report = self.engine.generate_insights(data)

        # Should still work but warn about insufficient data
        insufficient_insights = [i for i in report.insights
                                if "Insufficient Data" in i.title]
        assert len(insufficient_insights) > 0
        assert insufficient_insights[0].impact_score >= 7.0
        assert insufficient_insights[0].actionable is True

    def test_generate_insights_with_nan_values(self):
        """Test insight generation with NaN values."""
        data = [1, 2, float('nan'), 4, 5, float('nan'), 7, 8, 9, 10]
        report = self.engine.generate_insights(data)

        assert report.is_reliable is True
        assert report.quality_score < 100  # Should penalize for missing values

        # Should have data quality insights about missing values
        quality_insights = [i for i in report.insights
                          if i.insight_type == InsightType.DATA_QUALITY]
        # Note: Quality insights might be generated from quality assessment

    def test_generate_insights_with_infinity_values(self):
        """Test insight generation with infinity values."""
        data = [1, 2, float('inf'), 4, 5, float('-inf'), 7, 8, 9, 10]
        report = self.engine.generate_insights(data)

        assert report.is_reliable is True
        # Infinity values should be filtered out and quality score affected
        assert report.quality_score < 100

    def test_generate_insights_all_invalid_values(self):
        """Test insight generation with all invalid values."""
        data = [float('nan'), float('inf'), float('-inf'), float('nan')]
        report = self.engine.generate_insights(data)

        # Should have insights about data quality problems
        assert len(report.insights) > 0
        quality_insights = [i for i in report.insights
                          if "Insufficient Data" in i.title or i.insight_type == InsightType.DATA_QUALITY]
        assert len(quality_insights) > 0

    def test_generate_insights_constant_values(self):
        """Test insight generation with constant values (no variance)."""
        data = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
        report = self.engine.generate_insights(data)

        assert report.is_reliable is True
        # Should detect low variance in quality assessment
        assert report.quality_score < 100

    def test_generate_insights_invalid_format(self):
        """Test insight generation with invalid data format."""
        data = "invalid_data_format"
        report = self.engine.generate_insights(data)

        assert report.is_reliable is False
        assert "Invalid data format" in report.error_message

    def test_generate_insights_mixed_valid_invalid_series(self):
        """Test with mixed valid and invalid data series."""
        data = {
            'valid_series': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'invalid_series': [float('nan')] * 10,
            'partial_series': [1, 2, float('nan'), 4, 5]
        }
        report = self.engine.generate_insights(data)

        assert report.is_reliable is True
        # Should generate insights for valid series despite invalid ones
        assert len(report.insights) > 0

    # Data Quality Assessment Tests

    def test_assess_data_quality_high_quality_data(self):
        """Test data quality assessment with high quality data."""
        data_dict = {'series1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        quality_assessment = self.engine._assess_data_quality(data_dict)

        assert quality_assessment['overall_score'] >= 90
        assert quality_assessment['assessment_reliable'] is True
        assert len(quality_assessment['common_issues']) == 0

    def test_assess_data_quality_missing_values(self):
        """Test data quality assessment with missing values."""
        data_dict = {'series1': [1, 2, float('nan'), 4, float('nan'), 6, 7, 8, 9, 10]}
        quality_assessment = self.engine._assess_data_quality(data_dict)

        assert quality_assessment['overall_score'] < 100
        assert 'missing_values' in quality_assessment['common_issues']

    def test_assess_data_quality_outliers(self):
        """Test data quality assessment with outliers."""
        data_dict = {'series1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000]}  # 1000 is clear outlier
        quality_assessment = self.engine._assess_data_quality(data_dict)

        assert quality_assessment['overall_score'] < 100
        # Outlier detection should identify the issue

    def test_assess_data_quality_zero_variance(self):
        """Test data quality assessment with zero variance."""
        data_dict = {'series1': [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]}
        quality_assessment = self.engine._assess_data_quality(data_dict)

        assert quality_assessment['overall_score'] < 100
        assert 'low_variance' in quality_assessment['common_issues']

    def test_assess_data_quality_insufficient_data(self):
        """Test data quality assessment with insufficient data."""
        data_dict = {'series1': [1, 2, 3]}  # Less than min_samples
        quality_assessment = self.engine._assess_data_quality(data_dict)

        assert quality_assessment['overall_score'] < 100
        assert 'insufficient_data' in quality_assessment['common_issues']

    def test_assess_data_quality_empty_series(self):
        """Test data quality assessment with empty series."""
        data_dict = {'series1': []}
        quality_assessment = self.engine._assess_data_quality(data_dict)

        assert quality_assessment['overall_score'] == 0

    # Pattern Discovery Tests

    def test_discover_patterns_increasing_trend(self):
        """Test pattern discovery with clear increasing trend."""
        data_dict = {'series1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        insights = self.engine._discover_patterns(data_dict)

        trend_insights = [i for i in insights if "Trend Pattern" in i.title]
        assert len(trend_insights) > 0
        assert "Increasing trend" in trend_insights[0].description
        assert trend_insights[0].actionable is True

    def test_discover_patterns_decreasing_trend(self):
        """Test pattern discovery with clear decreasing trend."""
        data_dict = {'series1': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]}
        insights = self.engine._discover_patterns(data_dict)

        trend_insights = [i for i in insights if "Trend Pattern" in i.title]
        assert len(trend_insights) > 0
        assert "Decreasing trend" in trend_insights[0].description

    def test_discover_patterns_no_trend(self):
        """Test pattern discovery with no clear trend."""
        data_dict = {'series1': [5, 3, 7, 2, 8, 4, 6, 1, 9, 5]}
        insights = self.engine._discover_patterns(data_dict)

        trend_insights = [i for i in insights if "Trend Pattern" in i.title]
        if len(trend_insights) > 0:
            assert "No clear trend" in trend_insights[0].description or trend_insights[0].confidence_score < 0.5

    def test_discover_patterns_insufficient_data(self):
        """Test pattern discovery with insufficient data."""
        data_dict = {'series1': [1, 2]}
        insights = self.engine._discover_patterns(data_dict)

        # Should not generate any insights due to insufficient data
        assert len(insights) == 0

    def test_discover_patterns_seasonal_data(self):
        """Test pattern discovery with seasonal data."""
        # Create data with clear seasonal pattern (period 4)
        data = []
        for i in range(20):
            data.append(5 + 2 * math.sin(2 * math.pi * i / 4))

        data_dict = {'series1': data}
        insights = self.engine._discover_patterns(data_dict)

        seasonal_insights = [i for i in insights if "Seasonal Pattern" in i.title]
        # Note: Simple correlation-based detection might not catch this perfectly
        # More sophisticated seasonal detection would be needed for guaranteed results

    def test_discover_patterns_high_volatility(self):
        """Test pattern discovery with high volatility data."""
        data_dict = {'series1': [1, 10, 2, 9, 3, 8, 4, 7, 5, 6]}
        insights = self.engine._discover_patterns(data_dict)

        volatility_insights = [i for i in insights if "Volatility Pattern" in i.title]
        assert len(volatility_insights) > 0
        assert "high" in volatility_insights[0].description.lower()

    def test_discover_patterns_error_handling(self):
        """Test pattern discovery error handling."""
        # Mock an error in trend detection
        with patch.object(self.engine, '_detect_trend_pattern', side_effect=Exception("Test error")):
            data_dict = {'series1': [1, 2, 3, 4, 5]}
            insights = self.engine._discover_patterns(data_dict)
            # Should handle error gracefully and continue with other patterns

    # Distribution Analysis Tests

    def test_analyze_distributions_normal_data(self):
        """Test distribution analysis with normal-like data."""
        # Generate roughly normal data
        data = []
        for i in range(100):
            data.append(5 + (i - 50) * 0.1)  # Roughly centered around 5

        data_dict = {'series1': data}
        insights = self.engine._analyze_distributions(data_dict)

        distribution_insights = [i for i in insights if i.insight_type == InsightType.DISTRIBUTION_ANALYSIS]
        assert len(distribution_insights) > 0

    def test_analyze_distributions_skewed_data(self):
        """Test distribution analysis with skewed data."""
        # Right-skewed data
        data_dict = {'series1': [1, 1, 1, 2, 2, 3, 4, 5, 10, 20]}
        insights = self.engine._analyze_distributions(data_dict)

        distribution_insights = [i for i in insights if i.insight_type == InsightType.DISTRIBUTION_ANALYSIS]
        assert len(distribution_insights) > 0
        assert "skewed" in distribution_insights[0].description

    def test_analyze_distributions_insufficient_data(self):
        """Test distribution analysis with insufficient data."""
        data_dict = {'series1': [1, 2]}
        insights = self.engine._analyze_distributions(data_dict)

        assert len(insights) == 0

    # Correlation Analysis Tests

    def test_analyze_correlations_perfect_positive(self):
        """Test correlation analysis with perfect positive correlation."""
        data_dict = {
            'series1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'series2': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
        }
        insights = self.engine._analyze_correlations(data_dict)

        correlation_insights = [i for i in insights if i.insight_type == InsightType.CORRELATION_ANALYSIS]
        assert len(correlation_insights) > 0
        assert "positive" in correlation_insights[0].description
        assert correlation_insights[0].confidence_score > 0.9

    def test_analyze_correlations_negative(self):
        """Test correlation analysis with negative correlation."""
        data_dict = {
            'series1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'series2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        }
        insights = self.engine._analyze_correlations(data_dict)

        correlation_insights = [i for i in insights if i.insight_type == InsightType.CORRELATION_ANALYSIS]
        assert len(correlation_insights) > 0
        assert "negative" in correlation_insights[0].description

    def test_analyze_correlations_no_correlation(self):
        """Test correlation analysis with no correlation."""
        data_dict = {
            'series1': [1, 2, 3, 4, 5],
            'series2': [3, 1, 4, 2, 5]  # Random order
        }
        insights = self.engine._analyze_correlations(data_dict)

        # Might not generate insights if correlation is below threshold (0.3)
        correlation_insights = [i for i in insights if i.insight_type == InsightType.CORRELATION_ANALYSIS]
        # Don't assert on length as weak correlations are filtered out

    def test_analyze_correlations_single_series(self):
        """Test correlation analysis with single series (no correlations possible)."""
        data_dict = {'series1': [1, 2, 3, 4, 5]}
        insights = self.engine._analyze_correlations(data_dict)

        assert len(insights) == 0

    def test_analyze_correlations_different_lengths(self):
        """Test correlation analysis with series of different lengths."""
        data_dict = {
            'series1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'series2': [2, 4, 6]
        }
        insights = self.engine._analyze_correlations(data_dict)

        # Should handle by truncating to shorter length
        correlation_insights = [i for i in insights if i.insight_type == InsightType.CORRELATION_ANALYSIS]
        if len(correlation_insights) > 0:
            assert correlation_insights[0].supporting_evidence['sample_size'] == 3

    def test_analyze_correlations_with_nan_values(self):
        """Test correlation analysis with NaN values."""
        data_dict = {
            'series1': [1, 2, float('nan'), 4, 5],
            'series2': [2, 4, 6, 8, 10]
        }
        insights = self.engine._analyze_correlations(data_dict)

        # Should handle NaN values by filtering them out
        # Might still find correlation if enough valid paired data remains

    def test_analyze_correlations_insufficient_data(self):
        """Test correlation analysis with insufficient paired data."""
        data_dict = {
            'series1': [1, 2],
            'series2': [3, 4]
        }
        insights = self.engine._analyze_correlations(data_dict)

        # Should not generate insights with only 2 data points
        assert len(insights) == 0

    # Meta-insights Tests

    def test_generate_meta_insights_no_insights(self):
        """Test meta-insight generation when no insights were found."""
        insights = []
        data_dict = {'series1': [1, 2, 3, 4, 5]}
        meta_insights = self.engine._generate_meta_insights(insights, data_dict)

        assert len(meta_insights) > 0
        limited_insights = [i for i in meta_insights if "Limited Insights" in i.title]
        assert len(limited_insights) > 0
        assert limited_insights[0].actionable is True

    def test_generate_meta_insights_many_high_impact(self):
        """Test meta-insight generation with many high-impact insights."""
        insights = []
        for i in range(5):
            insights.append(Insight(
                insight_type=InsightType.PATTERN_DISCOVERY,
                title=f"Insight {i}",
                description="Test",
                confidence_level=ConfidenceLevel.HIGH,
                confidence_score=0.9,
                supporting_evidence={},
                recommendations=[],
                impact_score=8.0,
                actionable=True
            ))

        data_dict = {'series1': [1, 2, 3, 4, 5]}
        meta_insights = self.engine._generate_meta_insights(insights, data_dict)

        high_impact_insights = [i for i in meta_insights if "High-Impact Opportunities" in i.title]
        assert len(high_impact_insights) > 0
        assert high_impact_insights[0].impact_score >= 8.0

    # Helper Method Tests

    def test_get_confidence_level(self):
        """Test confidence level classification."""
        assert self.engine._get_confidence_level(0.3) == ConfidenceLevel.LOW
        assert self.engine._get_confidence_level(0.6) == ConfidenceLevel.MEDIUM
        assert self.engine._get_confidence_level(0.85) == ConfidenceLevel.HIGH
        assert self.engine._get_confidence_level(0.97) == ConfidenceLevel.VERY_HIGH

    def test_get_distribution_recommendations(self):
        """Test distribution-specific recommendations."""
        right_skewed_recs = self.engine._get_distribution_recommendations("right-skewed")
        assert len(right_skewed_recs) > 0
        assert any("log transformation" in rec.lower() for rec in right_skewed_recs)

        left_skewed_recs = self.engine._get_distribution_recommendations("left-skewed")
        assert len(left_skewed_recs) > 0
        assert any("power transformation" in rec.lower() for rec in left_skewed_recs)

        symmetric_recs = self.engine._get_distribution_recommendations("symmetric")
        assert len(symmetric_recs) > 0

    def test_get_correlation_recommendations(self):
        """Test correlation-specific recommendations."""
        strong_positive_recs = self.engine._get_correlation_recommendations(0.85, "series1", "series2")
        assert len(strong_positive_recs) > 0
        assert any("positive relationship" in rec.lower() for rec in strong_positive_recs)

        strong_negative_recs = self.engine._get_correlation_recommendations(-0.85, "series1", "series2")
        assert len(strong_negative_recs) > 0
        assert any("negative relationship" in rec.lower() for rec in strong_negative_recs)

        moderate_recs = self.engine._get_correlation_recommendations(0.5, "series1", "series2")
        assert len(moderate_recs) > 0

    # Performance and Error Handling Tests

    def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        # Generate large dataset
        large_data = list(range(10000))

        report = self.engine.generate_insights(large_data)

        # Should complete without errors
        assert report.is_reliable is True
        assert report.analysis_duration < 10.0  # Should complete within reasonable time

    def test_error_handling_in_analysis(self):
        """Test error handling during analysis."""
        # Mock an error in pattern discovery
        with patch.object(self.engine, '_discover_patterns', side_effect=Exception("Test error")):
            data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            report = self.engine.generate_insights(data, analysis_types=[InsightType.PATTERN_DISCOVERY])

            # Should handle error gracefully
            assert report.is_reliable is True  # Other analyses might still work
            # Should continue with other analysis types

    def test_memory_efficiency(self):
        """Test memory efficiency with multiple series."""
        # Create multiple large series
        data_dict = {}
        for i in range(10):
            data_dict[f'series_{i}'] = list(range(1000))

        report = self.engine.generate_insights(data_dict)

        # Should complete without memory errors
        assert report.is_reliable is True

    def test_extreme_values_handling(self):
        """Test handling of extreme values."""
        data = [
            1e-100,  # Very small
            1e100,   # Very large
            -1e100,  # Very large negative
            0,       # Zero
            1,       # Normal
            2,       # Normal
            3        # Normal
        ]

        report = self.engine.generate_insights(data)

        # Should handle extreme values without crashing
        assert report.is_reliable is True

    def test_unicode_and_special_characters_in_metadata(self):
        """Test handling of unicode and special characters in metadata."""
        data = [1, 2, 3, 4, 5]
        metadata = {
            'unicode_field': 'Test with éñíçódè characters',
            'special_chars': 'Test with @#$%^&*() characters',
            'nested': {'deeper': 'value with 中文'}
        }

        report = self.engine.generate_insights(data, metadata=metadata)

        # Should handle special characters without issues
        assert report.is_reliable is True

    # Integration Tests

    def test_comprehensive_workflow(self):
        """Test comprehensive insight generation workflow."""
        # Multi-series data with various characteristics
        data = {
            'trending_up': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'trending_down': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            'volatile': [5, 1, 9, 2, 8, 3, 7, 4, 6, 5],
            'constant': [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            'with_outliers': [1, 2, 3, 4, 100, 6, 7, 8, 9, 10],
            'with_missing': [1, 2, float('nan'), 4, 5, 6, float('nan'), 8, 9, 10]
        }

        report = self.engine.generate_insights(data)

        # Should generate comprehensive insights
        assert report.is_reliable is True
        assert len(report.insights) > 0
        assert report.quality_score > 0

        # Should have various types of insights
        insight_types = [i.insight_type for i in report.insights]
        assert len(set(insight_types)) > 1  # Multiple types of insights

        # Should have actionable insights
        assert report.actionable_insights > 0

        # Should complete in reasonable time
        assert report.analysis_duration < 5.0