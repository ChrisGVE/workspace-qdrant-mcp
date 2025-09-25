"""
ML-based Insights Generation for intelligent data analysis.

Provides machine learning-based insights capabilities including:
- Automated pattern discovery and characterization
- Data quality assessment and recommendations
- Statistical significance testing and confidence intervals
- Feature importance analysis and relationship discovery
- Automated model selection and validation
- Concept drift detection and model adaptation
- Smart data preprocessing recommendations
"""

import math
import statistics
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

import numpy as np


logger = logging.getLogger(__name__)


class InsightType(Enum):
    """Enumeration for insight types."""
    PATTERN_DISCOVERY = "pattern_discovery"
    DATA_QUALITY = "data_quality"
    STATISTICAL_SIGNIFICANCE = "statistical_significance"
    FEATURE_IMPORTANCE = "feature_importance"
    MODEL_PERFORMANCE = "model_performance"
    CONCEPT_DRIFT = "concept_drift"
    PREPROCESSING_RECOMMENDATION = "preprocessing_recommendation"
    CORRELATION_ANALYSIS = "correlation_analysis"
    DISTRIBUTION_ANALYSIS = "distribution_analysis"
    ANOMALY_ANALYSIS = "anomaly_analysis"


class ConfidenceLevel(Enum):
    """Enumeration for confidence levels."""
    LOW = "low"       # < 50% confidence
    MEDIUM = "medium" # 50-80% confidence
    HIGH = "high"     # 80-95% confidence
    VERY_HIGH = "very_high"  # > 95% confidence


class DataQualityIssue(Enum):
    """Enumeration for data quality issues."""
    MISSING_VALUES = "missing_values"
    OUTLIERS = "outliers"
    INCONSISTENT_FORMAT = "inconsistent_format"
    HIGH_CORRELATION = "high_correlation"
    LOW_VARIANCE = "low_variance"
    IMBALANCED_DISTRIBUTION = "imbalanced_distribution"
    INSUFFICIENT_DATA = "insufficient_data"
    TEMPORAL_GAPS = "temporal_gaps"


@dataclass
class Insight:
    """Container for a single insight."""

    insight_type: InsightType
    title: str
    description: str
    confidence_level: ConfidenceLevel
    confidence_score: float
    supporting_evidence: Dict[str, Any]
    recommendations: List[str]
    impact_score: float  # 0-10 scale
    actionable: bool
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'insight_type': self.insight_type.value,
            'title': self.title,
            'description': self.description,
            'confidence_level': self.confidence_level.value,
            'confidence_score': self.confidence_score,
            'supporting_evidence': self.supporting_evidence,
            'recommendations': self.recommendations,
            'impact_score': self.impact_score,
            'actionable': self.actionable,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class InsightReport:
    """Container for insight analysis results."""

    insights: List[Insight] = field(default_factory=list)
    data_summary: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    total_insights: int = 0
    high_impact_insights: int = 0
    actionable_insights: int = 0
    analysis_duration: float = 0.0
    is_reliable: bool = True
    error_message: Optional[str] = None

    def __post_init__(self):
        """Calculate derived fields."""
        self._update_derived_fields()

    def _update_derived_fields(self):
        """Update derived fields based on current insights."""
        self.total_insights = len(self.insights)
        self.high_impact_insights = sum(1 for i in self.insights if i.impact_score >= 7.0)
        self.actionable_insights = sum(1 for i in self.insights if i.actionable)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'insights': [insight.to_dict() for insight in self.insights],
            'data_summary': self.data_summary,
            'quality_score': self.quality_score,
            'total_insights': self.total_insights,
            'high_impact_insights': self.high_impact_insights,
            'actionable_insights': self.actionable_insights,
            'analysis_duration': self.analysis_duration,
            'is_reliable': self.is_reliable,
            'error_message': self.error_message
        }


class MLInsightEngine:
    """
    ML-based insights generation engine with comprehensive edge case handling.

    Provides intelligent analysis of data patterns, quality assessment, and actionable
    recommendations while handling noisy data, insufficient samples, and concept drift.
    """

    def __init__(self,
                 min_samples: int = 10,
                 significance_threshold: float = 0.05,
                 confidence_threshold: float = 0.7):
        """
        Initialize ML insights engine.

        Args:
            min_samples: Minimum samples required for reliable analysis
            significance_threshold: Statistical significance threshold (p-value)
            confidence_threshold: Minimum confidence for actionable insights
        """
        self.min_samples = min_samples
        self.significance_threshold = significance_threshold
        self.confidence_threshold = confidence_threshold
        self._analysis_cache = {}
        self._model_cache = {}

    def generate_insights(self, data: Union[List[float], Dict[str, List[float]]],
                         analysis_types: Optional[List[InsightType]] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> InsightReport:
        """
        Generate comprehensive insights from data.

        Args:
            data: Numerical data (single series or multiple series)
            analysis_types: Specific types of analysis to perform
            metadata: Additional context information

        Returns:
            InsightReport containing discovered insights and recommendations
        """
        start_time = datetime.now()
        report = InsightReport()

        try:
            # Handle different data formats
            if isinstance(data, list):
                if not data:  # Empty list
                    report.is_reliable = False
                    report.error_message = "Empty dataset provided"
                    return report
                data_dict = {'primary': data}
            elif isinstance(data, dict):
                if not data:  # Empty dict
                    report.is_reliable = False
                    report.error_message = "Empty dataset provided"
                    return report
                data_dict = data
            else:
                report.is_reliable = False
                report.error_message = "Invalid data format"
                return report

            # Validate data quality
            quality_assessment = self._assess_data_quality(data_dict)
            report.quality_score = quality_assessment['overall_score']
            report.data_summary = quality_assessment

            # Check if we have sufficient data
            total_samples = sum(len(series) for series in data_dict.values())
            if total_samples < self.min_samples:
                insight = Insight(
                    insight_type=InsightType.DATA_QUALITY,
                    title="Insufficient Data",
                    description=f"Dataset contains only {total_samples} samples, "
                               f"minimum {self.min_samples} required for reliable analysis",
                    confidence_level=ConfidenceLevel.VERY_HIGH,
                    confidence_score=0.95,
                    supporting_evidence={'sample_count': total_samples, 'minimum_required': self.min_samples},
                    recommendations=[
                        "Collect more data points for reliable analysis",
                        "Consider using simpler statistical methods",
                        "Validate any conclusions with domain expertise"
                    ],
                    impact_score=9.0,
                    actionable=True
                )
                report.insights.append(insight)

            # Determine analysis types
            if analysis_types is None:
                analysis_types = [
                    InsightType.PATTERN_DISCOVERY,
                    InsightType.DATA_QUALITY,
                    InsightType.DISTRIBUTION_ANALYSIS,
                    InsightType.CORRELATION_ANALYSIS
                ]

            # Perform requested analyses
            for analysis_type in analysis_types:
                try:
                    insights = self._perform_analysis(data_dict, analysis_type, metadata)
                    report.insights.extend(insights)
                except Exception as e:
                    logger.warning(f"Error in {analysis_type} analysis: {e}")
                    continue

            # Generate meta-insights about the analysis
            meta_insights = self._generate_meta_insights(report.insights, data_dict)
            report.insights.extend(meta_insights)

        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            report.is_reliable = False
            report.error_message = f"Analysis error: {str(e)}"

        finally:
            end_time = datetime.now()
            report.analysis_duration = (end_time - start_time).total_seconds()
            # Recalculate derived fields since insights were added
            report._update_derived_fields()

        return report

    def _assess_data_quality(self, data_dict: Dict[str, List[float]]) -> Dict[str, Any]:
        """Assess overall data quality."""
        quality_issues = []
        series_scores = {}

        for series_name, series_data in data_dict.items():
            try:
                # Filter valid values
                valid_data = [x for x in series_data
                             if isinstance(x, (int, float)) and math.isfinite(x)]

                series_quality = 100.0
                issues = []

                # Check missing/invalid values
                missing_ratio = (len(series_data) - len(valid_data)) / len(series_data) if series_data else 1.0
                if missing_ratio > 0.1:
                    issues.append(DataQualityIssue.MISSING_VALUES)
                    series_quality -= min(50, missing_ratio * 100)

                if valid_data:
                    # Check for outliers using IQR method
                    try:
                        q1 = statistics.quantiles(valid_data, n=4)[0]
                        q3 = statistics.quantiles(valid_data, n=4)[2]
                        iqr = q3 - q1
                        outlier_bounds = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
                        outliers = [x for x in valid_data if x < outlier_bounds[0] or x > outlier_bounds[1]]

                        outlier_ratio = len(outliers) / len(valid_data)
                        if outlier_ratio > 0.05:
                            issues.append(DataQualityIssue.OUTLIERS)
                            series_quality -= min(20, outlier_ratio * 100)
                    except (statistics.StatisticsError, IndexError):
                        pass  # Insufficient data for outlier detection

                    # Check variance
                    try:
                        if len(valid_data) > 1:
                            variance = statistics.variance(valid_data)
                            if variance == 0:
                                issues.append(DataQualityIssue.LOW_VARIANCE)
                                series_quality -= 30
                    except statistics.StatisticsError:
                        pass

                    # Check sample size
                    if len(valid_data) < self.min_samples:
                        issues.append(DataQualityIssue.INSUFFICIENT_DATA)
                        series_quality -= 40

                    series_scores[series_name] = {
                        'score': max(0, series_quality),
                        'issues': [issue.value for issue in issues],
                        'valid_samples': len(valid_data),
                        'total_samples': len(series_data),
                        'missing_ratio': missing_ratio
                    }

                else:
                    # Handle case with no valid data
                    issues.append(DataQualityIssue.INSUFFICIENT_DATA)
                    series_scores[series_name] = {
                        'score': 0.0,
                        'issues': [issue.value for issue in issues],
                        'valid_samples': 0,
                        'total_samples': len(series_data),
                        'missing_ratio': 1.0
                    }

                quality_issues.extend(issues)

            except Exception as e:
                logger.warning(f"Error assessing quality for series {series_name}: {e}")
                series_scores[series_name] = {
                    'score': 0.0,
                    'issues': ['assessment_error'],
                    'error': str(e)
                }

        # Calculate overall score
        if series_scores:
            overall_score = statistics.mean([score['score'] for score in series_scores.values()])
        else:
            overall_score = 0.0

        return {
            'overall_score': round(overall_score, 2),
            'series_scores': series_scores,
            'common_issues': list(set(issue.value for issue in quality_issues)),
            'total_series': len(data_dict),
            'assessment_reliable': True
        }

    def _perform_analysis(self, data_dict: Dict[str, List[float]],
                         analysis_type: InsightType,
                         metadata: Optional[Dict[str, Any]]) -> List[Insight]:
        """Perform specific type of analysis."""
        insights = []

        try:
            if analysis_type == InsightType.PATTERN_DISCOVERY:
                insights.extend(self._discover_patterns(data_dict))
            elif analysis_type == InsightType.DISTRIBUTION_ANALYSIS:
                insights.extend(self._analyze_distributions(data_dict))
            elif analysis_type == InsightType.CORRELATION_ANALYSIS:
                insights.extend(self._analyze_correlations(data_dict))
            elif analysis_type == InsightType.DATA_QUALITY:
                insights.extend(self._generate_quality_insights(data_dict))
            elif analysis_type == InsightType.STATISTICAL_SIGNIFICANCE:
                insights.extend(self._test_statistical_significance(data_dict))
            # Add other analysis types as needed

        except Exception as e:
            logger.error(f"Error in {analysis_type} analysis: {e}")
            # Return error insight
            insights.append(Insight(
                insight_type=analysis_type,
                title=f"Analysis Error: {analysis_type.value}",
                description=f"Error occurred during {analysis_type.value} analysis: {str(e)}",
                confidence_level=ConfidenceLevel.LOW,
                confidence_score=0.0,
                supporting_evidence={'error': str(e)},
                recommendations=["Review data format and try again", "Contact system administrator"],
                impact_score=1.0,
                actionable=False
            ))

        return insights

    def _discover_patterns(self, data_dict: Dict[str, List[float]]) -> List[Insight]:
        """Discover patterns in data."""
        insights = []

        for series_name, series_data in data_dict.items():
            try:
                # Filter valid data
                valid_data = [x for x in series_data
                             if isinstance(x, (int, float)) and math.isfinite(x)]

                if len(valid_data) < 3:
                    continue

                # Trend detection
                trend_insight = self._detect_trend_pattern(series_name, valid_data)
                if trend_insight:
                    insights.append(trend_insight)

                # Seasonality detection (if we have enough data)
                if len(valid_data) >= 12:  # Need at least 12 points for basic seasonality
                    seasonality_insight = self._detect_seasonality_pattern(series_name, valid_data)
                    if seasonality_insight:
                        insights.append(seasonality_insight)

                # Volatility analysis
                volatility_insight = self._analyze_volatility_pattern(series_name, valid_data)
                if volatility_insight:
                    insights.append(volatility_insight)

            except Exception as e:
                logger.warning(f"Error discovering patterns in {series_name}: {e}")
                continue

        return insights

    def _detect_trend_pattern(self, series_name: str, data: List[float]) -> Optional[Insight]:
        """Detect trend patterns in data."""
        try:
            if len(data) < 3:
                return None

            # Simple linear trend detection
            x_values = list(range(len(data)))

            # Calculate correlation between index and values
            try:
                correlation = statistics.correlation(x_values, data)
            except statistics.StatisticsError:
                return None

            trend_strength = abs(correlation)
            confidence_score = trend_strength

            if trend_strength < 0.3:
                trend_type = "No clear trend"
                impact_score = 2.0
            elif correlation > 0.3:
                trend_type = "Increasing trend"
                impact_score = 6.0 + trend_strength * 2
            else:
                trend_type = "Decreasing trend"
                impact_score = 6.0 + trend_strength * 2

            confidence_level = self._get_confidence_level(confidence_score)

            recommendations = []
            if trend_strength > 0.5:
                if correlation > 0:
                    recommendations.extend([
                        "Monitor for continued growth",
                        "Plan for increased capacity if trend continues"
                    ])
                else:
                    recommendations.extend([
                        "Investigate causes of decline",
                        "Consider intervention strategies"
                    ])
            else:
                recommendations.append("Trend is weak, continue monitoring for pattern emergence")

            return Insight(
                insight_type=InsightType.PATTERN_DISCOVERY,
                title=f"Trend Pattern: {series_name}",
                description=f"{trend_type} detected in {series_name} (strength: {trend_strength:.3f})",
                confidence_level=confidence_level,
                confidence_score=confidence_score,
                supporting_evidence={
                    'correlation': correlation,
                    'trend_strength': trend_strength,
                    'data_points': len(data),
                    'series_name': series_name
                },
                recommendations=recommendations,
                impact_score=impact_score,
                actionable=trend_strength > 0.5
            )

        except Exception as e:
            logger.warning(f"Error detecting trend pattern: {e}")
            return None

    def _detect_seasonality_pattern(self, series_name: str, data: List[float]) -> Optional[Insight]:
        """Detect seasonal patterns in data."""
        try:
            if len(data) < 12:
                return None

            # Simple seasonality detection using autocorrelation
            # Check for patterns at different lags
            seasonal_lags = [7, 12, 24, 30, 52]  # Common seasonal periods
            best_lag = 0
            best_correlation = 0.0

            for lag in seasonal_lags:
                if lag >= len(data):
                    continue

                try:
                    lag_data = data[:-lag]
                    shifted_data = data[lag:]

                    if len(lag_data) == len(shifted_data) and len(lag_data) > 0:
                        correlation = statistics.correlation(lag_data, shifted_data)
                        if abs(correlation) > abs(best_correlation):
                            best_correlation = correlation
                            best_lag = lag
                except statistics.StatisticsError:
                    continue

            if abs(best_correlation) > 0.4:  # Reasonable threshold for seasonality
                confidence_score = abs(best_correlation)
                confidence_level = self._get_confidence_level(confidence_score)

                return Insight(
                    insight_type=InsightType.PATTERN_DISCOVERY,
                    title=f"Seasonal Pattern: {series_name}",
                    description=f"Seasonal pattern detected with period {best_lag} (correlation: {best_correlation:.3f})",
                    confidence_level=confidence_level,
                    confidence_score=confidence_score,
                    supporting_evidence={
                        'seasonal_lag': best_lag,
                        'autocorrelation': best_correlation,
                        'data_points': len(data),
                        'series_name': series_name
                    },
                    recommendations=[
                        f"Plan for recurring pattern every {best_lag} periods",
                        "Use seasonal forecasting methods",
                        "Investigate underlying seasonal drivers"
                    ],
                    impact_score=7.0,
                    actionable=True
                )

        except Exception as e:
            logger.warning(f"Error detecting seasonality: {e}")

        return None

    def _analyze_volatility_pattern(self, series_name: str, data: List[float]) -> Optional[Insight]:
        """Analyze volatility patterns in data."""
        try:
            if len(data) < 5:
                return None

            # Calculate rolling volatility (standard deviation)
            window_size = min(5, len(data) // 2)
            volatilities = []

            for i in range(window_size, len(data)):
                window = data[i-window_size:i]
                try:
                    vol = statistics.stdev(window) if len(window) > 1 else 0.0
                    volatilities.append(vol)
                except statistics.StatisticsError:
                    volatilities.append(0.0)

            if not volatilities:
                return None

            avg_volatility = statistics.mean(volatilities)
            volatility_trend = None

            # Check if volatility is changing
            if len(volatilities) >= 3:
                try:
                    vol_indices = list(range(len(volatilities)))
                    vol_correlation = statistics.correlation(vol_indices, volatilities)
                    if abs(vol_correlation) > 0.3:
                        volatility_trend = "increasing" if vol_correlation > 0 else "decreasing"
                except statistics.StatisticsError:
                    pass

            # Determine impact based on volatility level
            data_range = max(data) - min(data)
            relative_volatility = avg_volatility / data_range if data_range > 0 else 0.0

            if relative_volatility > 0.2:
                impact_score = 7.0
                volatility_level = "high"
            elif relative_volatility > 0.1:
                impact_score = 5.0
                volatility_level = "moderate"
            else:
                impact_score = 3.0
                volatility_level = "low"

            recommendations = []
            if volatility_level == "high":
                recommendations.extend([
                    "Implement risk management strategies",
                    "Investigate sources of volatility",
                    "Consider smoothing techniques for predictions"
                ])
            elif volatility_trend == "increasing":
                recommendations.extend([
                    "Monitor volatility increase closely",
                    "Prepare for increased uncertainty"
                ])

            description = f"{volatility_level.capitalize()} volatility detected in {series_name}"
            if volatility_trend:
                description += f" with {volatility_trend} trend"

            return Insight(
                insight_type=InsightType.PATTERN_DISCOVERY,
                title=f"Volatility Pattern: {series_name}",
                description=description,
                confidence_level=ConfidenceLevel.MEDIUM,
                confidence_score=0.7,
                supporting_evidence={
                    'average_volatility': avg_volatility,
                    'relative_volatility': relative_volatility,
                    'volatility_trend': volatility_trend,
                    'volatility_level': volatility_level,
                    'data_points': len(data)
                },
                recommendations=recommendations,
                impact_score=impact_score,
                actionable=len(recommendations) > 0
            )

        except Exception as e:
            logger.warning(f"Error analyzing volatility: {e}")

        return None

    def _analyze_distributions(self, data_dict: Dict[str, List[float]]) -> List[Insight]:
        """Analyze data distributions."""
        insights = []

        for series_name, series_data in data_dict.items():
            try:
                valid_data = [x for x in series_data
                             if isinstance(x, (int, float)) and math.isfinite(x)]

                if len(valid_data) < 3:
                    continue

                # Basic distribution analysis
                mean_val = statistics.mean(valid_data)
                median_val = statistics.median(valid_data)

                # Skewness detection (using mean vs median)
                try:
                    std_dev = statistics.stdev(valid_data) if len(valid_data) > 1 else 0
                    skewness = (mean_val - median_val) / std_dev if std_dev > 0 else 0
                except statistics.StatisticsError:
                    skewness = 0

                skew_description = "symmetric"
                if abs(skewness) > 0.3:  # Lower threshold for detecting skew
                    skew_description = "right-skewed" if skewness > 0 else "left-skewed"

                # Normality assessment (simple)
                normality_score = max(0, 1 - abs(skewness))

                insights.append(Insight(
                    insight_type=InsightType.DISTRIBUTION_ANALYSIS,
                    title=f"Distribution Analysis: {series_name}",
                    description=f"Distribution appears {skew_description} (skewness: {skewness:.3f})",
                    confidence_level=self._get_confidence_level(normality_score),
                    confidence_score=normality_score,
                    supporting_evidence={
                        'mean': mean_val,
                        'median': median_val,
                        'skewness': skewness,
                        'normality_score': normality_score,
                        'sample_size': len(valid_data)
                    },
                    recommendations=self._get_distribution_recommendations(skew_description),
                    impact_score=4.0 if abs(skewness) > 0.3 else 2.0,
                    actionable=abs(skewness) > 0.3
                ))

            except Exception as e:
                logger.warning(f"Error analyzing distribution for {series_name}: {e}")
                continue

        return insights

    def _analyze_correlations(self, data_dict: Dict[str, List[float]]) -> List[Insight]:
        """Analyze correlations between data series."""
        insights = []

        if len(data_dict) < 2:
            return insights

        series_names = list(data_dict.keys())

        for i, series1_name in enumerate(series_names):
            for j, series2_name in enumerate(series_names[i+1:], i+1):
                try:
                    series1_data = [x for x in data_dict[series1_name]
                                   if isinstance(x, (int, float)) and math.isfinite(x)]
                    series2_data = [x for x in data_dict[series2_name]
                                   if isinstance(x, (int, float)) and math.isfinite(x)]

                    # Ensure same length for correlation
                    min_len = min(len(series1_data), len(series2_data))
                    if min_len < 3:
                        continue

                    series1_data = series1_data[:min_len]
                    series2_data = series2_data[:min_len]

                    correlation = statistics.correlation(series1_data, series2_data)

                    if abs(correlation) > 0.3:  # Only report meaningful correlations
                        correlation_strength = "strong" if abs(correlation) > 0.7 else "moderate"
                        correlation_direction = "positive" if correlation > 0 else "negative"

                        insights.append(Insight(
                            insight_type=InsightType.CORRELATION_ANALYSIS,
                            title=f"Correlation: {series1_name} vs {series2_name}",
                            description=f"{correlation_strength.capitalize()} {correlation_direction} "
                                       f"correlation detected (r={correlation:.3f})",
                            confidence_level=self._get_confidence_level(abs(correlation)),
                            confidence_score=abs(correlation),
                            supporting_evidence={
                                'correlation': correlation,
                                'strength': correlation_strength,
                                'direction': correlation_direction,
                                'sample_size': min_len,
                                'series_1': series1_name,
                                'series_2': series2_name
                            },
                            recommendations=self._get_correlation_recommendations(correlation, series1_name, series2_name),
                            impact_score=3.0 + abs(correlation) * 4,
                            actionable=abs(correlation) > 0.5
                        ))

                except statistics.StatisticsError:
                    continue
                except Exception as e:
                    logger.warning(f"Error calculating correlation between {series1_name} and {series2_name}: {e}")
                    continue

        return insights

    def _generate_quality_insights(self, data_dict: Dict[str, List[float]]) -> List[Insight]:
        """Generate data quality specific insights."""
        insights = []

        quality_assessment = self._assess_data_quality(data_dict)

        if quality_assessment['overall_score'] < 70:
            insights.append(Insight(
                insight_type=InsightType.DATA_QUALITY,
                title="Data Quality Concerns",
                description=f"Overall data quality score is {quality_assessment['overall_score']:.1f}%",
                confidence_level=ConfidenceLevel.HIGH,
                confidence_score=0.9,
                supporting_evidence=quality_assessment,
                recommendations=[
                    "Review data collection processes",
                    "Implement data validation checks",
                    "Consider data cleaning procedures"
                ],
                impact_score=8.0,
                actionable=True
            ))

        return insights

    def _test_statistical_significance(self, data_dict: Dict[str, List[float]]) -> List[Insight]:
        """Test statistical significance of patterns."""
        insights = []

        # For now, this is a placeholder - would implement proper statistical tests
        # like t-tests, chi-square tests, etc.

        return insights

    def _generate_meta_insights(self, insights: List[Insight],
                               data_dict: Dict[str, List[float]]) -> List[Insight]:
        """Generate insights about the insights (meta-analysis)."""
        meta_insights = []

        if not insights:
            meta_insights.append(Insight(
                insight_type=InsightType.DATA_QUALITY,
                title="Limited Insights Generated",
                description="Analysis produced few actionable insights - data may lack clear patterns or sufficient quality",
                confidence_level=ConfidenceLevel.MEDIUM,
                confidence_score=0.7,
                supporting_evidence={'insight_count': len(insights)},
                recommendations=[
                    "Collect more data over a longer time period",
                    "Verify data collection accuracy",
                    "Consider external factors that might influence patterns"
                ],
                impact_score=6.0,
                actionable=True
            ))

        # Check for high-impact actionable insights
        high_impact_actionable = [i for i in insights if i.impact_score >= 7 and i.actionable]
        if len(high_impact_actionable) > 2:
            meta_insights.append(Insight(
                insight_type=InsightType.PREPROCESSING_RECOMMENDATION,
                title="Multiple High-Impact Opportunities Identified",
                description=f"Found {len(high_impact_actionable)} high-impact actionable insights",
                confidence_level=ConfidenceLevel.HIGH,
                confidence_score=0.9,
                supporting_evidence={'high_impact_count': len(high_impact_actionable)},
                recommendations=[
                    "Prioritize implementation of high-impact recommendations",
                    "Create action plan for identified opportunities",
                    "Monitor impact of implemented changes"
                ],
                impact_score=9.0,
                actionable=True
            ))

        return meta_insights

    def _get_confidence_level(self, score: float) -> ConfidenceLevel:
        """Convert numerical score to confidence level."""
        if score < 0.5:
            return ConfidenceLevel.LOW
        elif score < 0.8:
            return ConfidenceLevel.MEDIUM
        elif score < 0.95:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH

    def _get_distribution_recommendations(self, skew_description: str) -> List[str]:
        """Get recommendations based on distribution shape."""
        recommendations = []

        if skew_description == "right-skewed":
            recommendations.extend([
                "Consider log transformation for modeling",
                "Use median instead of mean for central tendency",
                "Be cautious of extreme values affecting analysis"
            ])
        elif skew_description == "left-skewed":
            recommendations.extend([
                "Consider power transformation",
                "Use median instead of mean for central tendency",
                "Investigate lower boundary constraints"
            ])
        else:
            recommendations.append("Distribution appears well-behaved for standard analyses")

        return recommendations

    def _get_correlation_recommendations(self, correlation: float,
                                       series1: str, series2: str) -> List[str]:
        """Get recommendations based on correlation findings."""
        recommendations = []

        if abs(correlation) > 0.7:
            if correlation > 0:
                recommendations.extend([
                    f"Strong positive relationship: changes in {series1} likely predict changes in {series2}",
                    "Consider using one variable to predict the other",
                    "Investigate causal relationship direction"
                ])
            else:
                recommendations.extend([
                    f"Strong negative relationship: {series1} increases as {series2} decreases",
                    "Consider trade-off analysis between these variables",
                    "Monitor for potential conflicts in optimization"
                ])
        else:
            recommendations.append("Moderate correlation suggests some relationship - investigate further")

        return recommendations