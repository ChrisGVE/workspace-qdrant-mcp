"""
Advanced Test Result Aggregation and Analytics System

This module provides comprehensive analytics capabilities for test execution results,
including trend analysis, performance metrics, failure pattern detection, and
detailed reporting with visualization support.

Features:
- Real-time test result aggregation and processing
- Historical trend analysis and performance tracking
- Flaky test detection with statistical analysis
- Test suite health monitoring and alerting
- Performance regression detection
- Code coverage analysis and reporting
- Detailed failure categorization and root cause analysis
- Export capabilities for CI/CD integration
"""

import json
import logging
import math
import sqlite3
import statistics
import threading
import time
from collections import defaultdict, Counter, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Dict, List, Set, Optional, Tuple, Any, Union,
    Callable, NamedTuple
)

# Use basic statistics instead of numpy/scipy for better compatibility
import math

from .discovery import TestMetadata, TestCategory, TestComplexity
from .execution import ExecutionResult, ExecutionStatus


class HealthStatus(Enum):
    """Test suite health status levels."""
    EXCELLENT = auto()      # >95% pass rate, fast execution, stable
    GOOD = auto()          # >90% pass rate, reasonable performance
    FAIR = auto()          # >80% pass rate, some issues detected
    POOR = auto()          # >70% pass rate, significant issues
    CRITICAL = auto()      # <=70% pass rate, major problems


class TrendDirection(Enum):
    """Trend direction indicators."""
    IMPROVING = auto()     # Performance/reliability getting better
    STABLE = auto()        # No significant change
    DEGRADING = auto()     # Performance/reliability getting worse
    VOLATILE = auto()      # Inconsistent results


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = auto()          # Informational alerts
    WARNING = auto()       # Warning conditions
    ERROR = auto()         # Error conditions requiring attention
    CRITICAL = auto()      # Critical issues requiring immediate action


@dataclass
class TestMetrics:
    """Comprehensive metrics for a single test."""
    test_name: str
    total_runs: int = 0
    passed_runs: int = 0
    failed_runs: int = 0
    skipped_runs: int = 0
    timeout_runs: int = 0

    # Performance metrics
    min_duration: float = float('inf')
    max_duration: float = 0.0
    avg_duration: float = 0.0
    duration_std: float = 0.0

    # Reliability metrics
    success_rate: float = 0.0
    flakiness_score: float = 0.0
    stability_index: float = 0.0

    # Trend metrics
    recent_success_rate: float = 0.0
    performance_trend: TrendDirection = TrendDirection.STABLE
    reliability_trend: TrendDirection = TrendDirection.STABLE

    # Failure analysis
    failure_patterns: Dict[str, int] = field(default_factory=dict)
    common_errors: List[str] = field(default_factory=list)

    # Historical data (limited to recent entries)
    duration_history: deque = field(default_factory=lambda: deque(maxlen=50))
    result_history: deque = field(default_factory=lambda: deque(maxlen=100))

    last_updated: float = field(default_factory=time.time)


@dataclass
class SuiteMetrics:
    """Comprehensive metrics for the entire test suite."""
    total_tests: int = 0
    executed_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    timeout_tests: int = 0

    # Performance metrics
    total_duration: float = 0.0
    avg_test_duration: float = 0.0
    suite_efficiency: float = 0.0  # tests per minute

    # Quality metrics
    overall_success_rate: float = 0.0
    health_status: HealthStatus = HealthStatus.EXCELLENT
    flaky_test_count: int = 0

    # Coverage metrics (if available)
    code_coverage: Optional[float] = None
    branch_coverage: Optional[float] = None
    coverage_trend: TrendDirection = TrendDirection.STABLE

    # Distribution by category
    category_breakdown: Dict[TestCategory, int] = field(default_factory=dict)
    complexity_breakdown: Dict[TestComplexity, int] = field(default_factory=dict)

    # Trends and alerts
    performance_trend: TrendDirection = TrendDirection.STABLE
    reliability_trend: TrendDirection = TrendDirection.STABLE
    alerts: List[str] = field(default_factory=list)

    timestamp: float = field(default_factory=time.time)


@dataclass
class Alert:
    """Test suite alert information."""
    level: AlertLevel
    message: str
    test_name: Optional[str] = None
    category: Optional[str] = None
    threshold_value: Optional[float] = None
    actual_value: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False


class FlakeDetector:
    """Advanced flaky test detection using statistical analysis."""

    def __init__(self, min_runs: int = 10, significance_level: float = 0.05):
        """
        Initialize flake detector.

        Args:
            min_runs: Minimum number of runs to consider for flake detection
            significance_level: Statistical significance level for detection
        """
        self.min_runs = min_runs
        self.significance_level = significance_level

    def calculate_flakiness_score(self, result_history: List[bool]) -> float:
        """
        Calculate flakiness score based on result pattern.

        Args:
            result_history: List of boolean results (True=pass, False=fail)

        Returns:
            Flakiness score between 0.0 (stable) and 1.0 (very flaky)
        """
        if len(result_history) < self.min_runs:
            return 0.0

        # Convert to binary list
        results = [1 if r else 0 for r in result_history]

        # Calculate basic statistics
        success_rate = sum(results) / len(results)

        # If all pass or all fail, it's stable (not flaky)
        if success_rate == 0.0 or success_rate == 1.0:
            return 0.0

        # Calculate run-length statistics
        runs = self._calculate_runs(results)
        expected_runs = 2 * len(results) * success_rate * (1 - success_rate)

        if expected_runs == 0:
            return 0.0

        # Z-score for runs test
        z_runs = abs(runs - expected_runs) / math.sqrt(expected_runs)

        # Calculate switching frequency
        switches = sum(1 for i in range(1, len(results))
                      if results[i] != results[i-1])
        switch_rate = switches / (len(results) - 1) if len(results) > 1 else 0

        # Combine metrics for flakiness score
        # Higher switching rate and unusual run patterns indicate flakiness
        flakiness_score = min(1.0, (switch_rate * 2 + z_runs / 10) *
                             (1 - abs(success_rate - 0.5) * 2))

        return flakiness_score

    def _calculate_runs(self, binary_list: List[int]) -> int:
        """Calculate the number of runs in a binary sequence."""
        if len(binary_list) == 0:
            return 0

        runs = 1
        for i in range(1, len(binary_list)):
            if binary_list[i] != binary_list[i-1]:
                runs += 1

        return runs

    def detect_flaky_tests(self, metrics: Dict[str, TestMetrics]) -> List[str]:
        """
        Detect flaky tests based on statistical analysis.

        Args:
            metrics: Dictionary of test metrics

        Returns:
            List of test names identified as flaky
        """
        flaky_tests = []

        for test_name, test_metrics in metrics.items():
            if test_metrics.total_runs < self.min_runs:
                continue

            # Use result history for flakiness calculation
            result_history = [r == ExecutionStatus.COMPLETED
                            for r in test_metrics.result_history]

            flakiness_score = self.calculate_flakiness_score(result_history)
            test_metrics.flakiness_score = flakiness_score

            # Consider test flaky if score is above threshold and has mixed results
            if (flakiness_score > 0.3 and
                0.1 < test_metrics.success_rate < 0.9):
                flaky_tests.append(test_name)

        return flaky_tests


class TrendAnalyzer:
    """Analyzes trends in test performance and reliability."""

    def __init__(self, window_size: int = 20):
        """
        Initialize trend analyzer.

        Args:
            window_size: Number of recent data points to consider for trend analysis
        """
        self.window_size = window_size

    def analyze_performance_trend(self, duration_history: List[float]) -> TrendDirection:
        """
        Analyze performance trend based on execution duration history.

        Args:
            duration_history: List of execution durations

        Returns:
            Trend direction
        """
        if len(duration_history) < 10:
            return TrendDirection.STABLE

        # Use recent window
        recent_data = list(duration_history)[-self.window_size:]

        if len(recent_data) < 5:
            return TrendDirection.STABLE

        # Simple linear trend calculation
        try:
            n = len(recent_data)
            x_sum = sum(range(n))
            y_sum = sum(recent_data)
            xy_sum = sum(i * recent_data[i] for i in range(n))
            x_squared_sum = sum(i * i for i in range(n))

            # Calculate slope using least squares
            denominator = n * x_squared_sum - x_sum * x_sum
            if denominator == 0:
                return TrendDirection.STABLE

            slope = (n * xy_sum - x_sum * y_sum) / denominator

            # Calculate relative change
            avg_duration = sum(recent_data) / len(recent_data)
            relative_slope = slope / avg_duration if avg_duration > 0 else 0

            if abs(relative_slope) < 0.1:  # Less than 10% change per unit time
                return TrendDirection.STABLE
            elif relative_slope > 0.1:
                return TrendDirection.DEGRADING  # Getting slower
            else:
                return TrendDirection.IMPROVING  # Getting faster

        except Exception:
            return TrendDirection.STABLE

    def analyze_reliability_trend(self, result_history: List[bool]) -> TrendDirection:
        """
        Analyze reliability trend based on pass/fail history.

        Args:
            result_history: List of boolean results (True=pass, False=fail)

        Returns:
            Trend direction
        """
        if len(result_history) < 10:
            return TrendDirection.STABLE

        # Use recent window
        recent_results = result_history[-self.window_size:]

        if len(recent_results) < 5:
            return TrendDirection.STABLE

        # Calculate moving averages for trend detection
        window_size = min(5, len(recent_results) // 2)

        early_window = recent_results[:window_size]
        late_window = recent_results[-window_size:]

        early_rate = sum(early_window) / len(early_window)
        late_rate = sum(late_window) / len(late_window)

        rate_change = late_rate - early_rate

        # Determine trend based on rate change
        if abs(rate_change) < 0.1:  # Less than 10% change
            # Check for volatility
            volatility = self._calculate_volatility(recent_results)
            if volatility > 0.4:
                return TrendDirection.VOLATILE
            else:
                return TrendDirection.STABLE
        elif rate_change > 0.1:
            return TrendDirection.IMPROVING
        else:
            return TrendDirection.DEGRADING

    def _calculate_volatility(self, result_history: List[bool]) -> float:
        """Calculate volatility of test results."""
        if len(result_history) < 2:
            return 0.0

        # Calculate switching rate as a measure of volatility
        switches = sum(1 for i in range(1, len(result_history))
                      if result_history[i] != result_history[i-1])

        return switches / (len(result_history) - 1)


class AlertManager:
    """Manages test suite alerts and notifications."""

    def __init__(self):
        self.alerts: List[Alert] = []
        self._alert_thresholds = {
            'success_rate_warning': 0.90,
            'success_rate_critical': 0.70,
            'flakiness_warning': 0.3,
            'flakiness_critical': 0.6,
            'performance_degradation': 0.2,  # 20% performance degradation
            'timeout_rate_warning': 0.05,   # 5% timeout rate
        }

    def check_suite_health(self, suite_metrics: SuiteMetrics) -> List[Alert]:
        """
        Check overall suite health and generate alerts.

        Args:
            suite_metrics: Suite-level metrics

        Returns:
            List of generated alerts
        """
        new_alerts = []

        # Success rate alerts
        if suite_metrics.overall_success_rate < self._alert_thresholds['success_rate_critical']:
            new_alerts.append(Alert(
                level=AlertLevel.CRITICAL,
                message=f"Critical: Suite success rate is {suite_metrics.overall_success_rate:.1%}",
                category="reliability",
                threshold_value=self._alert_thresholds['success_rate_critical'],
                actual_value=suite_metrics.overall_success_rate
            ))
        elif suite_metrics.overall_success_rate < self._alert_thresholds['success_rate_warning']:
            new_alerts.append(Alert(
                level=AlertLevel.WARNING,
                message=f"Warning: Suite success rate is {suite_metrics.overall_success_rate:.1%}",
                category="reliability",
                threshold_value=self._alert_thresholds['success_rate_warning'],
                actual_value=suite_metrics.overall_success_rate
            ))

        # Flaky test alerts
        if suite_metrics.flaky_test_count > 0:
            flaky_ratio = suite_metrics.flaky_test_count / suite_metrics.total_tests
            if flaky_ratio > 0.1:  # More than 10% flaky tests
                new_alerts.append(Alert(
                    level=AlertLevel.ERROR,
                    message=f"High flaky test count: {suite_metrics.flaky_test_count} tests ({flaky_ratio:.1%})",
                    category="flakiness",
                    actual_value=flaky_ratio
                ))
            elif flaky_ratio > 0.05:  # More than 5% flaky tests
                new_alerts.append(Alert(
                    level=AlertLevel.WARNING,
                    message=f"Elevated flaky test count: {suite_metrics.flaky_test_count} tests ({flaky_ratio:.1%})",
                    category="flakiness",
                    actual_value=flaky_ratio
                ))

        # Performance alerts
        if suite_metrics.performance_trend == TrendDirection.DEGRADING:
            new_alerts.append(Alert(
                level=AlertLevel.WARNING,
                message="Performance degradation detected across test suite",
                category="performance"
            ))

        # Reliability trend alerts
        if suite_metrics.reliability_trend == TrendDirection.DEGRADING:
            new_alerts.append(Alert(
                level=AlertLevel.ERROR,
                message="Reliability degradation detected across test suite",
                category="reliability"
            ))
        elif suite_metrics.reliability_trend == TrendDirection.VOLATILE:
            new_alerts.append(Alert(
                level=AlertLevel.WARNING,
                message="Volatile test results detected - investigate for intermittent issues",
                category="stability"
            ))

        self.alerts.extend(new_alerts)
        return new_alerts

    def check_test_health(self, test_metrics: TestMetrics) -> List[Alert]:
        """
        Check individual test health and generate alerts.

        Args:
            test_metrics: Test-specific metrics

        Returns:
            List of generated alerts
        """
        new_alerts = []

        # Flakiness alerts
        if test_metrics.flakiness_score > self._alert_thresholds['flakiness_critical']:
            new_alerts.append(Alert(
                level=AlertLevel.CRITICAL,
                message=f"Critical flaky test detected",
                test_name=test_metrics.test_name,
                category="flakiness",
                threshold_value=self._alert_thresholds['flakiness_critical'],
                actual_value=test_metrics.flakiness_score
            ))
        elif test_metrics.flakiness_score > self._alert_thresholds['flakiness_warning']:
            new_alerts.append(Alert(
                level=AlertLevel.WARNING,
                message=f"Flaky test detected",
                test_name=test_metrics.test_name,
                category="flakiness",
                threshold_value=self._alert_thresholds['flakiness_warning'],
                actual_value=test_metrics.flakiness_score
            ))

        # Performance degradation alerts
        if test_metrics.performance_trend == TrendDirection.DEGRADING:
            new_alerts.append(Alert(
                level=AlertLevel.WARNING,
                message=f"Performance degradation detected",
                test_name=test_metrics.test_name,
                category="performance"
            ))

        # High failure rate alerts
        if test_metrics.total_runs >= 10 and test_metrics.success_rate < 0.5:
            new_alerts.append(Alert(
                level=AlertLevel.ERROR,
                message=f"High failure rate: {test_metrics.success_rate:.1%}",
                test_name=test_metrics.test_name,
                category="reliability",
                actual_value=test_metrics.success_rate
            ))

        self.alerts.extend(new_alerts)
        return new_alerts


class TestAnalytics:
    """Comprehensive test analytics system."""

    def __init__(self, database_path: Optional[Path] = None):
        """
        Initialize test analytics system.

        Args:
            database_path: Path to SQLite database for persistent storage
        """
        self.database_path = database_path or Path(".test_analytics.db")
        self.test_metrics: Dict[str, TestMetrics] = {}
        self.suite_history: List[SuiteMetrics] = []

        self.flake_detector = FlakeDetector()
        self.trend_analyzer = TrendAnalyzer()
        self.alert_manager = AlertManager()

        self._analytics_lock = threading.RLock()

        # Initialize database
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database for analytics storage."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS test_results (
                        test_name TEXT,
                        timestamp REAL,
                        status TEXT,
                        duration REAL,
                        error_message TEXT,
                        PRIMARY KEY (test_name, timestamp)
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS suite_metrics (
                        timestamp REAL PRIMARY KEY,
                        total_tests INTEGER,
                        passed_tests INTEGER,
                        failed_tests INTEGER,
                        total_duration REAL,
                        success_rate REAL,
                        health_status TEXT
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL,
                        level TEXT,
                        message TEXT,
                        test_name TEXT,
                        category TEXT,
                        acknowledged INTEGER DEFAULT 0
                    )
                """)

        except Exception as e:
            logging.warning(f"Failed to initialize analytics database: {e}")

    def process_execution_results(self,
                                results: Dict[str, ExecutionResult],
                                test_metadata: Dict[str, TestMetadata]) -> SuiteMetrics:
        """
        Process test execution results and update analytics.

        Args:
            results: Test execution results
            test_metadata: Test metadata

        Returns:
            Updated suite metrics
        """
        with self._analytics_lock:
            # Update individual test metrics
            for test_name, result in results.items():
                self._update_test_metrics(test_name, result)
                self._store_test_result(test_name, result)

            # Calculate suite-level metrics
            suite_metrics = self._calculate_suite_metrics(results, test_metadata)

            # Analyze trends and detect issues
            self._analyze_trends()
            self._detect_flaky_tests()

            # Generate alerts
            self.alert_manager.check_suite_health(suite_metrics)
            for test_name, test_metrics in self.test_metrics.items():
                self.alert_manager.check_test_health(test_metrics)

            # Store suite metrics
            self.suite_history.append(suite_metrics)
            self._store_suite_metrics(suite_metrics)

            return suite_metrics

    def _update_test_metrics(self, test_name: str, result: ExecutionResult):
        """Update metrics for a single test."""
        if test_name not in self.test_metrics:
            self.test_metrics[test_name] = TestMetrics(test_name=test_name)

        metrics = self.test_metrics[test_name]

        # Update run counts
        metrics.total_runs += 1
        if result.status == ExecutionStatus.COMPLETED:
            metrics.passed_runs += 1
        elif result.status == ExecutionStatus.FAILED:
            metrics.failed_runs += 1

            # Analyze failure pattern
            if result.error_message:
                # Extract error type
                error_type = self._extract_error_type(result.error_message)
                metrics.failure_patterns[error_type] = metrics.failure_patterns.get(error_type, 0) + 1
        elif result.status == ExecutionStatus.TIMEOUT:
            metrics.timeout_runs += 1
        elif result.status == ExecutionStatus.SKIPPED:
            metrics.skipped_runs += 1

        # Update duration metrics
        if result.duration > 0:
            metrics.duration_history.append(result.duration)
            metrics.min_duration = min(metrics.min_duration, result.duration)
            metrics.max_duration = max(metrics.max_duration, result.duration)

            durations = list(metrics.duration_history)
            metrics.avg_duration = statistics.mean(durations)
            if len(durations) > 1:
                metrics.duration_std = statistics.stdev(durations)

        # Update result history
        metrics.result_history.append(result.status)

        # Calculate success rate
        if metrics.total_runs > 0:
            metrics.success_rate = metrics.passed_runs / metrics.total_runs

        # Calculate recent success rate (last 20 runs)
        recent_results = list(metrics.result_history)[-20:]
        if recent_results:
            recent_successes = sum(1 for r in recent_results if r == ExecutionStatus.COMPLETED)
            metrics.recent_success_rate = recent_successes / len(recent_results)

        metrics.last_updated = time.time()

    def _calculate_suite_metrics(self,
                                results: Dict[str, ExecutionResult],
                                test_metadata: Dict[str, TestMetadata]) -> SuiteMetrics:
        """Calculate suite-level metrics."""
        suite_metrics = SuiteMetrics()

        # Basic counts
        suite_metrics.total_tests = len(test_metadata)
        suite_metrics.executed_tests = len(results)

        status_counts = Counter(result.status for result in results.values())
        suite_metrics.passed_tests = status_counts.get(ExecutionStatus.COMPLETED, 0)
        suite_metrics.failed_tests = status_counts.get(ExecutionStatus.FAILED, 0)
        suite_metrics.skipped_tests = status_counts.get(ExecutionStatus.SKIPPED, 0)
        suite_metrics.timeout_tests = status_counts.get(ExecutionStatus.TIMEOUT, 0)

        # Performance metrics
        suite_metrics.total_duration = sum(r.duration for r in results.values())
        if results:
            suite_metrics.avg_test_duration = suite_metrics.total_duration / len(results)
            suite_metrics.suite_efficiency = len(results) / (suite_metrics.total_duration / 60) if suite_metrics.total_duration > 0 else 0

        # Quality metrics
        if suite_metrics.executed_tests > 0:
            suite_metrics.overall_success_rate = suite_metrics.passed_tests / suite_metrics.executed_tests

        # Health status
        suite_metrics.health_status = self._calculate_health_status(suite_metrics.overall_success_rate)

        # Category breakdown
        for metadata in test_metadata.values():
            suite_metrics.category_breakdown[metadata.category] = suite_metrics.category_breakdown.get(metadata.category, 0) + 1
            suite_metrics.complexity_breakdown[metadata.complexity] = suite_metrics.complexity_breakdown.get(metadata.complexity, 0) + 1

        # Flaky test count
        suite_metrics.flaky_test_count = len([
            name for name, metrics in self.test_metrics.items()
            if metrics.flakiness_score > 0.3
        ])

        # Trends
        if len(self.suite_history) >= 5:
            recent_rates = [s.overall_success_rate for s in self.suite_history[-5:]]
            suite_metrics.reliability_trend = self.trend_analyzer.analyze_reliability_trend(
                [rate > 0.8 for rate in recent_rates]
            )

            recent_durations = [s.avg_test_duration for s in self.suite_history[-5:]]
            suite_metrics.performance_trend = self.trend_analyzer.analyze_performance_trend(recent_durations)

        return suite_metrics

    def _calculate_health_status(self, success_rate: float) -> HealthStatus:
        """Calculate suite health status based on success rate."""
        if success_rate >= 0.95:
            return HealthStatus.EXCELLENT
        elif success_rate >= 0.90:
            return HealthStatus.GOOD
        elif success_rate >= 0.80:
            return HealthStatus.FAIR
        elif success_rate >= 0.70:
            return HealthStatus.POOR
        else:
            return HealthStatus.CRITICAL

    def _analyze_trends(self):
        """Analyze trends for all tests."""
        for test_name, metrics in self.test_metrics.items():
            if len(metrics.duration_history) > 0:
                metrics.performance_trend = self.trend_analyzer.analyze_performance_trend(
                    list(metrics.duration_history)
                )

            if len(metrics.result_history) > 0:
                result_bools = [r == ExecutionStatus.COMPLETED for r in metrics.result_history]
                metrics.reliability_trend = self.trend_analyzer.analyze_reliability_trend(result_bools)

    def _detect_flaky_tests(self):
        """Detect and update flaky test information."""
        flaky_tests = self.flake_detector.detect_flaky_tests(self.test_metrics)

        # Update flakiness scores in metrics (done by detector)
        # Additional processing could be added here

    def _extract_error_type(self, error_message: str) -> str:
        """Extract error type from error message."""
        if not error_message:
            return "Unknown"

        # Common error patterns
        error_patterns = {
            'AssertionError': 'AssertionError',
            'TimeoutError': 'TimeoutError',
            'ConnectionError': 'ConnectionError',
            'ImportError': 'ImportError',
            'AttributeError': 'AttributeError',
            'KeyError': 'KeyError',
            'ValueError': 'ValueError',
            'TypeError': 'TypeError',
        }

        for pattern, error_type in error_patterns.items():
            if pattern in error_message:
                return error_type

        # Extract first line as error type
        first_line = error_message.split('\n')[0].strip()
        return first_line[:50] if first_line else "Unknown"

    def _store_test_result(self, test_name: str, result: ExecutionResult):
        """Store test result in database."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO test_results
                    (test_name, timestamp, status, duration, error_message)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    test_name,
                    result.start_time,
                    result.status.name,
                    result.duration,
                    result.error_message
                ))
        except Exception as e:
            logging.warning(f"Failed to store test result: {e}")

    def _store_suite_metrics(self, suite_metrics: SuiteMetrics):
        """Store suite metrics in database."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO suite_metrics
                    (timestamp, total_tests, passed_tests, failed_tests,
                     total_duration, success_rate, health_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    suite_metrics.timestamp,
                    suite_metrics.total_tests,
                    suite_metrics.passed_tests,
                    suite_metrics.failed_tests,
                    suite_metrics.total_duration,
                    suite_metrics.overall_success_rate,
                    suite_metrics.health_status.name
                ))
        except Exception as e:
            logging.warning(f"Failed to store suite metrics: {e}")

    def get_test_report(self, test_name: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive report for a specific test."""
        if test_name not in self.test_metrics:
            return None

        metrics = self.test_metrics[test_name]

        return {
            'test_name': test_name,
            'metrics': asdict(metrics),
            'recent_failures': self._get_recent_failures(test_name, 5),
            'performance_statistics': self._get_performance_statistics(test_name),
            'recommendations': self._get_test_recommendations(metrics)
        }

    def get_suite_report(self) -> Dict[str, Any]:
        """Get comprehensive suite-level report."""
        if not self.suite_history:
            return {}

        latest_metrics = self.suite_history[-1]

        return {
            'current_metrics': asdict(latest_metrics),
            'trending_tests': self._get_trending_tests(),
            'flaky_tests': self._get_flaky_test_summary(),
            'performance_insights': self._get_performance_insights(),
            'recommendations': self._get_suite_recommendations(),
            'alerts': [asdict(alert) for alert in self.alert_manager.alerts[-10:]]  # Last 10 alerts
        }

    def _get_recent_failures(self, test_name: str, limit: int) -> List[Dict[str, Any]]:
        """Get recent failure information for a test."""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.execute("""
                    SELECT timestamp, error_message FROM test_results
                    WHERE test_name = ? AND status = 'FAILED'
                    ORDER BY timestamp DESC LIMIT ?
                """, (test_name, limit))

                return [
                    {
                        'timestamp': timestamp,
                        'error_message': error_message,
                        'date': datetime.fromtimestamp(timestamp).isoformat()
                    }
                    for timestamp, error_message in cursor.fetchall()
                ]
        except Exception:
            return []

    def _get_performance_statistics(self, test_name: str) -> Dict[str, Any]:
        """Get performance statistics for a test."""
        if test_name not in self.test_metrics:
            return {}

        metrics = self.test_metrics[test_name]
        durations = list(metrics.duration_history)

        if not durations:
            return {}

        return {
            'min_duration': min(durations),
            'max_duration': max(durations),
            'avg_duration': statistics.mean(durations),
            'median_duration': statistics.median(durations),
            'std_deviation': statistics.stdev(durations) if len(durations) > 1 else 0,
            'percentile_95': self._percentile(durations, 95) if durations else 0,
            'trend': metrics.performance_trend.name
        }

    def _get_test_recommendations(self, metrics: TestMetrics) -> List[str]:
        """Generate recommendations for a specific test."""
        recommendations = []

        if metrics.flakiness_score > 0.5:
            recommendations.append("Test is highly flaky - investigate for race conditions or dependencies")
        elif metrics.flakiness_score > 0.3:
            recommendations.append("Test shows signs of flakiness - consider adding waits or improving isolation")

        if metrics.performance_trend == TrendDirection.DEGRADING:
            recommendations.append("Performance is degrading - review for inefficient operations")

        if metrics.avg_duration > 30:  # 30 seconds
            recommendations.append("Test takes a long time to execute - consider optimization")

        if metrics.success_rate < 0.8:
            recommendations.append("Low success rate - investigate common failure causes")

        return recommendations

    def _get_trending_tests(self) -> Dict[str, List[str]]:
        """Get tests categorized by trends."""
        trends = {
            'improving': [],
            'degrading': [],
            'volatile': []
        }

        for test_name, metrics in self.test_metrics.items():
            if metrics.performance_trend == TrendDirection.IMPROVING or metrics.reliability_trend == TrendDirection.IMPROVING:
                trends['improving'].append(test_name)
            elif metrics.performance_trend == TrendDirection.DEGRADING or metrics.reliability_trend == TrendDirection.DEGRADING:
                trends['degrading'].append(test_name)
            elif metrics.reliability_trend == TrendDirection.VOLATILE:
                trends['volatile'].append(test_name)

        return trends

    def _get_flaky_test_summary(self) -> List[Dict[str, Any]]:
        """Get summary of flaky tests."""
        flaky_tests = []

        for test_name, metrics in self.test_metrics.items():
            if metrics.flakiness_score > 0.3:
                flaky_tests.append({
                    'test_name': test_name,
                    'flakiness_score': metrics.flakiness_score,
                    'success_rate': metrics.success_rate,
                    'total_runs': metrics.total_runs
                })

        return sorted(flaky_tests, key=lambda x: x['flakiness_score'], reverse=True)

    def _get_performance_insights(self) -> Dict[str, Any]:
        """Get performance insights for the suite."""
        if not self.suite_history:
            return {}

        recent_metrics = self.suite_history[-10:]  # Last 10 runs

        return {
            'average_suite_duration': statistics.mean([m.total_duration for m in recent_metrics]),
            'suite_efficiency_trend': [m.suite_efficiency for m in recent_metrics],
            'slowest_tests': self._get_slowest_tests(5),
            'performance_improvement_opportunities': self._identify_performance_opportunities()
        }

    def _get_slowest_tests(self, limit: int) -> List[Dict[str, Any]]:
        """Get the slowest tests."""
        test_durations = [
            {
                'test_name': name,
                'avg_duration': metrics.avg_duration,
                'max_duration': metrics.max_duration
            }
            for name, metrics in self.test_metrics.items()
            if metrics.avg_duration > 0
        ]

        return sorted(test_durations, key=lambda x: x['avg_duration'], reverse=True)[:limit]

    def _identify_performance_opportunities(self) -> List[str]:
        """Identify performance improvement opportunities."""
        opportunities = []

        # Find tests with high duration variance
        high_variance_tests = [
            name for name, metrics in self.test_metrics.items()
            if metrics.duration_std > metrics.avg_duration * 0.5  # High coefficient of variation
        ]

        if high_variance_tests:
            opportunities.append(f"Tests with inconsistent performance: {', '.join(high_variance_tests[:3])}")

        # Find slow test categories
        slow_categories = []
        category_durations = defaultdict(list)

        for name, metrics in self.test_metrics.items():
            if metrics.avg_duration > 0:
                # This would need test metadata to categorize properly
                category_durations['unknown'].append(metrics.avg_duration)

        return opportunities

    def _get_suite_recommendations(self) -> List[str]:
        """Generate suite-level recommendations."""
        recommendations = []

        if not self.suite_history:
            return recommendations

        latest = self.suite_history[-1]

        if latest.overall_success_rate < 0.9:
            recommendations.append("Suite success rate is below 90% - focus on improving test reliability")

        if latest.flaky_test_count > latest.total_tests * 0.1:
            recommendations.append("High number of flaky tests - implement better test isolation")

        if latest.performance_trend == TrendDirection.DEGRADING:
            recommendations.append("Performance is degrading - review recent changes and optimize slow tests")

        if latest.health_status in [HealthStatus.POOR, HealthStatus.CRITICAL]:
            recommendations.append("Suite health is poor - immediate attention required")

        return recommendations

    def export_data(self, format: str = 'json', file_path: Optional[Path] = None) -> Union[str, Path]:
        """
        Export analytics data in specified format.

        Args:
            format: Export format ('json', 'csv')
            file_path: Output file path

        Returns:
            Exported data as string or file path
        """
        if format.lower() == 'json':
            data = {
                'test_metrics': {name: asdict(metrics) for name, metrics in self.test_metrics.items()},
                'suite_history': [asdict(metrics) for metrics in self.suite_history],
                'alerts': [asdict(alert) for alert in self.alert_manager.alerts],
                'export_timestamp': time.time()
            }

            json_data = json.dumps(data, indent=2, default=str)

            if file_path:
                with open(file_path, 'w') as f:
                    f.write(json_data)
                return file_path
            else:
                return json_data
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0

        sorted_data = sorted(data)
        n = len(sorted_data)
        index = (percentile / 100) * (n - 1)

        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = int(index)
            upper = lower + 1
            weight = index - lower
            return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight

    def close(self):
        """Close the analytics system and cleanup resources."""
        # Database connections are handled per-operation, no cleanup needed
        pass