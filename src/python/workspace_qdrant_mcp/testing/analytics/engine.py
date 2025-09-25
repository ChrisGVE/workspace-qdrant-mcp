"""
Test Analytics Engine

Core analytics engine for test result processing, trend analysis,
and quality metrics calculation with comprehensive edge case handling
and performance monitoring.
"""

import logging
import sqlite3
import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Union, Tuple
from collections import defaultdict, Counter
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of test metrics."""
    PASS_RATE = "pass_rate"
    EXECUTION_TIME = "execution_time"
    COVERAGE = "coverage"
    FLAKINESS = "flakiness"
    COMPLEXITY = "complexity"
    MAINTAINABILITY = "maintainability"
    RELIABILITY = "reliability"


class TrendDirection(Enum):
    """Trend direction indicators."""
    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


@dataclass
class TestResult:
    """Represents a single test execution result."""
    test_id: str
    test_name: str
    file_path: Path
    status: str  # passed, failed, skipped, error
    execution_time: float
    timestamp: datetime
    error_message: Optional[str] = None
    coverage: Optional[float] = None
    memory_usage: Optional[int] = None
    retries: int = 0
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestMetrics:
    """Aggregated metrics for tests."""
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    pass_rate: float = 0.0
    avg_execution_time: float = 0.0
    total_execution_time: float = 0.0
    coverage_percentage: float = 0.0
    flakiness_score: float = 0.0
    reliability_score: float = 1.0
    trend: TrendDirection = TrendDirection.UNKNOWN
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None


@dataclass
class TrendAnalysis:
    """Trend analysis results."""
    metric_type: MetricType
    direction: TrendDirection
    change_percentage: float
    confidence: float
    data_points: List[Tuple[datetime, float]]
    regression_slope: Optional[float] = None
    correlation_coefficient: Optional[float] = None
    seasonal_pattern: Optional[Dict[str, Any]] = None
    anomalies: List[datetime] = field(default_factory=list)


@dataclass
class QualityReport:
    """Test quality assessment report."""
    overall_score: float  # 0-100
    metrics: Dict[MetricType, float]
    trends: List[TrendAnalysis]
    recommendations: List[str]
    warnings: List[str]
    critical_issues: List[str]
    generated_at: datetime = field(default_factory=datetime.now)


class TestAnalyticsEngine:
    """
    Core analytics engine for test result processing and analysis.

    Provides comprehensive test metrics calculation, trend analysis,
    quality scoring, and anomaly detection with robust error handling.
    """

    def __init__(self, db_path: Path, enable_ml_features: bool = True):
        """
        Initialize analytics engine.

        Args:
            db_path: Path to SQLite database for analytics data
            enable_ml_features: Enable machine learning features
        """
        self.db_path = db_path
        self.enable_ml_features = enable_ml_features
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Initialize analytics database schema."""
        conn = sqlite3.connect(self.db_path)
        try:
            # Test results table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS test_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id TEXT NOT NULL,
                    test_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    execution_time REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    error_message TEXT,
                    coverage REAL,
                    memory_usage INTEGER,
                    retries INTEGER DEFAULT 0,
                    tags TEXT,  -- JSON array
                    metadata TEXT,  -- JSON object
                    UNIQUE(test_id, timestamp)
                )
            ''')

            # Metrics snapshots table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS metrics_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_time REAL NOT NULL,
                    metric_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    tags TEXT,  -- JSON array
                    context TEXT  -- JSON object
                )
            ''')

            # Trend analysis cache
            conn.execute('''
                CREATE TABLE IF NOT EXISTS trend_cache (
                    metric_type TEXT PRIMARY KEY,
                    analysis_data TEXT NOT NULL,  -- JSON object
                    last_updated REAL NOT NULL
                )
            ''')

            # Quality reports
            conn.execute('''
                CREATE TABLE IF NOT EXISTS quality_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    generated_at REAL NOT NULL,
                    overall_score REAL NOT NULL,
                    report_data TEXT NOT NULL  -- JSON object
                )
            ''')

            # Create indexes for performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_test_results_timestamp ON test_results(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_test_results_test_id ON test_results(test_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_test_results_status ON test_results(status)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_metrics_snapshots_time ON metrics_snapshots(snapshot_time)')

            conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize analytics database: {e}")
            raise
        finally:
            conn.close()

    def record_test_result(self, result: TestResult) -> bool:
        """
        Record a test execution result.

        Args:
            result: Test result to record

        Returns:
            True if recorded successfully, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute('''
                    INSERT OR REPLACE INTO test_results (
                        test_id, test_name, file_path, status, execution_time,
                        timestamp, error_message, coverage, memory_usage, retries,
                        tags, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result.test_id,
                    result.test_name,
                    str(result.file_path),
                    result.status,
                    result.execution_time,
                    result.timestamp.timestamp(),
                    result.error_message,
                    result.coverage,
                    result.memory_usage,
                    result.retries,
                    json.dumps(list(result.tags)),
                    json.dumps(result.metadata)
                ))
                conn.commit()
                logger.debug(f"Recorded test result: {result.test_id}")
                return True
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"Failed to record test result {result.test_id}: {e}")
            return False

    def record_test_results_batch(self, results: List[TestResult]) -> int:
        """
        Record multiple test results in a batch.

        Args:
            results: List of test results to record

        Returns:
            Number of successfully recorded results
        """
        if not results:
            return 0

        recorded_count = 0
        conn = sqlite3.connect(self.db_path)

        try:
            # Prepare batch data
            batch_data = []
            for result in results:
                try:
                    batch_data.append((
                        result.test_id,
                        result.test_name,
                        str(result.file_path),
                        result.status,
                        result.execution_time,
                        result.timestamp.timestamp(),
                        result.error_message,
                        result.coverage,
                        result.memory_usage,
                        result.retries,
                        json.dumps(list(result.tags)),
                        json.dumps(result.metadata)
                    ))
                except Exception as e:
                    logger.warning(f"Failed to prepare result {result.test_id} for batch: {e}")

            if batch_data:
                conn.executemany('''
                    INSERT OR REPLACE INTO test_results (
                        test_id, test_name, file_path, status, execution_time,
                        timestamp, error_message, coverage, memory_usage, retries,
                        tags, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', batch_data)
                conn.commit()
                recorded_count = len(batch_data)

        except Exception as e:
            logger.error(f"Failed to record test results batch: {e}")
        finally:
            conn.close()

        logger.info(f"Recorded {recorded_count}/{len(results)} test results")
        return recorded_count

    def calculate_metrics(self,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         test_filter: Optional[str] = None) -> TestMetrics:
        """
        Calculate aggregated test metrics for a time period.

        Args:
            start_time: Start of analysis period
            end_time: End of analysis period
            test_filter: Optional SQL filter for test selection

        Returns:
            Calculated test metrics
        """
        end_time = end_time or datetime.now()
        start_time = start_time or (end_time - timedelta(days=7))

        conn = sqlite3.connect(self.db_path)
        try:
            # Base query
            base_query = '''
                SELECT status, execution_time, coverage
                FROM test_results
                WHERE timestamp >= ? AND timestamp <= ?
            '''
            params = [start_time.timestamp(), end_time.timestamp()]

            if test_filter:
                base_query += f" AND {test_filter}"

            cursor = conn.execute(base_query, params)
            results = cursor.fetchall()

            if not results:
                logger.warning(f"No test results found for period {start_time} to {end_time}")
                return TestMetrics(period_start=start_time, period_end=end_time)

            # Calculate basic metrics
            total_tests = len(results)
            status_counts = Counter(result[0] for result in results)
            execution_times = [result[1] for result in results if result[1] is not None]
            coverages = [result[2] for result in results if result[2] is not None]

            metrics = TestMetrics(
                total_tests=total_tests,
                passed=status_counts.get('passed', 0),
                failed=status_counts.get('failed', 0),
                skipped=status_counts.get('skipped', 0),
                errors=status_counts.get('error', 0),
                period_start=start_time,
                period_end=end_time
            )

            # Calculate derived metrics with error handling
            if total_tests > 0:
                metrics.pass_rate = (metrics.passed / total_tests) * 100

            if execution_times:
                metrics.avg_execution_time = statistics.mean(execution_times)
                metrics.total_execution_time = sum(execution_times)

            if coverages:
                metrics.coverage_percentage = statistics.mean(coverages)

            # Calculate flakiness score
            metrics.flakiness_score = self._calculate_flakiness_score(start_time, end_time)

            # Calculate reliability score
            metrics.reliability_score = self._calculate_reliability_score(metrics)

            # Determine trend
            metrics.trend = self._determine_trend(MetricType.PASS_RATE, end_time)

            logger.debug(f"Calculated metrics for {total_tests} tests")
            return metrics

        except Exception as e:
            logger.error(f"Failed to calculate metrics: {e}")
            return TestMetrics(period_start=start_time, period_end=end_time)
        finally:
            conn.close()

    def analyze_trends(self,
                      metric_type: MetricType,
                      days_back: int = 30,
                      min_data_points: int = 5) -> Optional[TrendAnalysis]:
        """
        Analyze trends for a specific metric.

        Args:
            metric_type: Type of metric to analyze
            days_back: Number of days to look back
            min_data_points: Minimum data points required for analysis

        Returns:
            Trend analysis results or None if insufficient data
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)

            # Get historical data points
            data_points = self._get_metric_history(metric_type, start_time, end_time)

            if len(data_points) < min_data_points:
                logger.warning(f"Insufficient data points ({len(data_points)}) for trend analysis of {metric_type}")
                return None

            # Calculate trend direction and statistics
            values = [point[1] for point in data_points]

            # Use linear regression for trend analysis
            trend_analysis = TrendAnalysis(
                metric_type=metric_type,
                data_points=data_points,
                direction=TrendDirection.STABLE,
                change_percentage=0.0,
                confidence=0.0
            )

            if len(values) >= 2:
                # Calculate slope using least squares
                x_values = list(range(len(values)))
                if len(set(values)) > 1:  # Avoid division by zero
                    slope = np.polyfit(x_values, values, 1)[0] if len(values) > 1 else 0
                    trend_analysis.regression_slope = slope

                    # Calculate correlation coefficient
                    if len(values) > 2:
                        correlation = np.corrcoef(x_values, values)[0, 1]
                        trend_analysis.correlation_coefficient = correlation
                        trend_analysis.confidence = abs(correlation)

                    # Determine trend direction
                    change_percentage = ((values[-1] - values[0]) / abs(values[0])) * 100 if values[0] != 0 else 0
                    trend_analysis.change_percentage = change_percentage

                    if abs(change_percentage) < 5:
                        trend_analysis.direction = TrendDirection.STABLE
                    elif change_percentage > 5:
                        trend_analysis.direction = TrendDirection.IMPROVING
                    elif change_percentage < -5:
                        trend_analysis.direction = TrendDirection.DECLINING

                    # Check for volatility
                    if len(values) > 3:
                        volatility = statistics.stdev(values) / statistics.mean(values) if statistics.mean(values) != 0 else 0
                        if volatility > 0.3:  # 30% coefficient of variation threshold
                            trend_analysis.direction = TrendDirection.VOLATILE

            # Detect anomalies if ML features enabled
            if self.enable_ml_features:
                trend_analysis.anomalies = self._detect_anomalies(data_points)

            logger.debug(f"Trend analysis complete for {metric_type}: {trend_analysis.direction.value}")
            return trend_analysis

        except Exception as e:
            logger.error(f"Failed to analyze trends for {metric_type}: {e}")
            return None

    def generate_quality_report(self, period_days: int = 7) -> QualityReport:
        """
        Generate comprehensive test quality report.

        Args:
            period_days: Number of days to analyze

        Returns:
            Quality assessment report
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=period_days)

            # Calculate current metrics
            current_metrics = self.calculate_metrics(start_time, end_time)

            # Analyze trends for key metrics
            trends = []
            key_metrics = [MetricType.PASS_RATE, MetricType.EXECUTION_TIME, MetricType.COVERAGE]

            for metric_type in key_metrics:
                trend = self.analyze_trends(metric_type, days_back=period_days * 2)
                if trend:
                    trends.append(trend)

            # Calculate quality scores
            quality_scores = self._calculate_quality_scores(current_metrics, trends)

            # Generate recommendations
            recommendations = self._generate_recommendations(current_metrics, trends)

            # Identify warnings and critical issues
            warnings, critical_issues = self._identify_issues(current_metrics, trends)

            # Calculate overall score
            overall_score = sum(quality_scores.values()) / len(quality_scores) if quality_scores else 0

            report = QualityReport(
                overall_score=overall_score,
                metrics=quality_scores,
                trends=trends,
                recommendations=recommendations,
                warnings=warnings,
                critical_issues=critical_issues
            )

            # Cache report
            self._cache_quality_report(report)

            logger.info(f"Generated quality report with score: {overall_score:.1f}")
            return report

        except Exception as e:
            logger.error(f"Failed to generate quality report: {e}")
            return QualityReport(
                overall_score=0,
                metrics={},
                trends=[],
                recommendations=["Error generating report - check logs"],
                warnings=[f"Report generation failed: {e}"],
                critical_issues=[]
            )

    def _calculate_flakiness_score(self, start_time: datetime, end_time: datetime) -> float:
        """Calculate test flakiness score."""
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                # Find tests with inconsistent results
                cursor = conn.execute('''
                    SELECT test_id, COUNT(DISTINCT status) as status_variations,
                           COUNT(*) as total_runs
                    FROM test_results
                    WHERE timestamp >= ? AND timestamp <= ?
                    GROUP BY test_id
                    HAVING total_runs > 1
                ''', (start_time.timestamp(), end_time.timestamp()))

                flaky_tests = 0
                total_tests = 0

                for test_id, variations, runs in cursor.fetchall():
                    total_tests += 1
                    if variations > 1:
                        flaky_tests += 1

                return (flaky_tests / total_tests) * 100 if total_tests > 0 else 0

            finally:
                conn.close()
        except Exception as e:
            logger.error(f"Failed to calculate flakiness score: {e}")
            return 0.0

    def _calculate_reliability_score(self, metrics: TestMetrics) -> float:
        """Calculate test reliability score."""
        try:
            # Base reliability on pass rate and consistency
            base_score = metrics.pass_rate / 100

            # Adjust for flakiness
            flakiness_penalty = metrics.flakiness_score / 100
            reliability = max(0, base_score - flakiness_penalty)

            return min(1.0, reliability)
        except Exception as e:
            logger.error(f"Failed to calculate reliability score: {e}")
            return 0.0

    def _determine_trend(self, metric_type: MetricType, end_time: datetime) -> TrendDirection:
        """Determine trend direction for a metric."""
        try:
            trend_analysis = self.analyze_trends(metric_type, days_back=14)
            return trend_analysis.direction if trend_analysis else TrendDirection.UNKNOWN
        except Exception as e:
            logger.error(f"Failed to determine trend: {e}")
            return TrendDirection.UNKNOWN

    def _get_metric_history(self, metric_type: MetricType, start_time: datetime, end_time: datetime) -> List[Tuple[datetime, float]]:
        """Get historical metric data points."""
        conn = sqlite3.connect(self.db_path)
        try:
            if metric_type == MetricType.PASS_RATE:
                # Calculate daily pass rates
                cursor = conn.execute('''
                    SELECT date(timestamp, 'unixepoch') as day,
                           AVG(CASE WHEN status = 'passed' THEN 100.0 ELSE 0.0 END) as pass_rate
                    FROM test_results
                    WHERE timestamp >= ? AND timestamp <= ?
                    GROUP BY day
                    ORDER BY day
                ''', (start_time.timestamp(), end_time.timestamp()))
            elif metric_type == MetricType.EXECUTION_TIME:
                # Calculate daily average execution time
                cursor = conn.execute('''
                    SELECT date(timestamp, 'unixepoch') as day,
                           AVG(execution_time) as avg_time
                    FROM test_results
                    WHERE timestamp >= ? AND timestamp <= ? AND execution_time IS NOT NULL
                    GROUP BY day
                    ORDER BY day
                ''', (start_time.timestamp(), end_time.timestamp()))
            else:
                return []

            data_points = []
            for row in cursor.fetchall():
                date_str, value = row
                if value is not None:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    data_points.append((date_obj, float(value)))

            return data_points
        finally:
            conn.close()

    def _detect_anomalies(self, data_points: List[Tuple[datetime, float]]) -> List[datetime]:
        """Detect anomalies in metric data using simple statistical methods."""
        if len(data_points) < 5:
            return []

        values = [point[1] for point in data_points]
        mean_val = statistics.mean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0

        if std_dev == 0:
            return []

        # Use 2-sigma rule for anomaly detection
        threshold = 2 * std_dev
        anomalies = []

        for timestamp, value in data_points:
            if abs(value - mean_val) > threshold:
                anomalies.append(timestamp)

        return anomalies

    def _calculate_quality_scores(self, metrics: TestMetrics, trends: List[TrendAnalysis]) -> Dict[MetricType, float]:
        """Calculate quality scores for different metrics."""
        scores = {}

        # Pass rate score (0-100)
        scores[MetricType.PASS_RATE] = metrics.pass_rate

        # Execution time score (inverse relationship)
        if metrics.avg_execution_time > 0:
            # Score decreases as execution time increases (normalized to 0-100)
            max_acceptable_time = 60.0  # 60 seconds
            time_score = max(0, 100 - (metrics.avg_execution_time / max_acceptable_time) * 100)
            scores[MetricType.EXECUTION_TIME] = min(100, time_score)

        # Coverage score
        scores[MetricType.COVERAGE] = metrics.coverage_percentage

        # Reliability score
        scores[MetricType.RELIABILITY] = metrics.reliability_score * 100

        # Flakiness score (inverse - lower is better)
        scores[MetricType.FLAKINESS] = max(0, 100 - metrics.flakiness_score)

        return scores

    def _generate_recommendations(self, metrics: TestMetrics, trends: List[TrendAnalysis]) -> List[str]:
        """Generate recommendations based on metrics and trends."""
        recommendations = []

        # Pass rate recommendations
        if metrics.pass_rate < 90:
            recommendations.append(f"Pass rate is {metrics.pass_rate:.1f}% - investigate failing tests")

        if metrics.pass_rate < 50:
            recommendations.append("Critical: Less than 50% pass rate - immediate investigation required")

        # Execution time recommendations
        if metrics.avg_execution_time > 30:
            recommendations.append(f"Average execution time is {metrics.avg_execution_time:.1f}s - consider optimization")

        # Flakiness recommendations
        if metrics.flakiness_score > 10:
            recommendations.append(f"Flakiness score is {metrics.flakiness_score:.1f}% - stabilize flaky tests")

        # Coverage recommendations
        if metrics.coverage_percentage < 80:
            recommendations.append(f"Coverage is {metrics.coverage_percentage:.1f}% - add more tests")

        # Trend-based recommendations
        for trend in trends:
            if trend.direction == TrendDirection.DECLINING and trend.confidence > 0.7:
                recommendations.append(f"Declining trend in {trend.metric_type.value} - investigate regression")

        if not recommendations:
            recommendations.append("Test suite quality looks good - maintain current practices")

        return recommendations

    def _identify_issues(self, metrics: TestMetrics, trends: List[TrendAnalysis]) -> Tuple[List[str], List[str]]:
        """Identify warnings and critical issues."""
        warnings = []
        critical_issues = []

        # Critical issues
        if metrics.pass_rate < 50:
            critical_issues.append(f"Critical low pass rate: {metrics.pass_rate:.1f}%")

        if metrics.flakiness_score > 30:
            critical_issues.append(f"High flakiness score: {metrics.flakiness_score:.1f}%")

        # Warnings
        if metrics.pass_rate < 80:
            warnings.append(f"Low pass rate: {metrics.pass_rate:.1f}%")

        if metrics.avg_execution_time > 60:
            warnings.append(f"Slow average execution time: {metrics.avg_execution_time:.1f}s")

        if metrics.coverage_percentage < 70:
            warnings.append(f"Low coverage: {metrics.coverage_percentage:.1f}%")

        for trend in trends:
            if trend.direction == TrendDirection.VOLATILE:
                warnings.append(f"Volatile {trend.metric_type.value} metric")

        return warnings, critical_issues

    def _cache_quality_report(self, report: QualityReport) -> None:
        """Cache quality report in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                # Serialize report data
                report_data = {
                    'overall_score': report.overall_score,
                    'metrics': {k.value: v for k, v in report.metrics.items()},
                    'trends_count': len(report.trends),
                    'recommendations_count': len(report.recommendations),
                    'warnings_count': len(report.warnings),
                    'critical_issues_count': len(report.critical_issues)
                }

                conn.execute('''
                    INSERT INTO quality_reports (generated_at, overall_score, report_data)
                    VALUES (?, ?, ?)
                ''', (report.generated_at.timestamp(), report.overall_score, json.dumps(report_data)))

                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"Failed to cache quality report: {e}")

    def get_dashboard_data(self, days_back: int = 7) -> Dict[str, Any]:
        """Get dashboard data for visualization."""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)

            # Get current metrics
            metrics = self.calculate_metrics(start_time, end_time)

            # Get trend data
            trends_data = {}
            for metric_type in [MetricType.PASS_RATE, MetricType.EXECUTION_TIME]:
                trend = self.analyze_trends(metric_type, days_back=days_back)
                if trend:
                    trends_data[metric_type.value] = {
                        'direction': trend.direction.value,
                        'change_percentage': trend.change_percentage,
                        'data_points': [(ts.isoformat(), val) for ts, val in trend.data_points]
                    }

            return {
                'metrics': {
                    'total_tests': metrics.total_tests,
                    'pass_rate': metrics.pass_rate,
                    'avg_execution_time': metrics.avg_execution_time,
                    'coverage_percentage': metrics.coverage_percentage,
                    'flakiness_score': metrics.flakiness_score
                },
                'trends': trends_data,
                'period': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'days': days_back
                }
            }

        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            return {'error': str(e)}