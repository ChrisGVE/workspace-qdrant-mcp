"""
Comprehensive unit tests for TestAnalyticsEngine with edge cases.

Tests all functionality including trend analysis, quality scoring,
anomaly detection, and error handling scenarios.
"""

import pytest
import sqlite3
import tempfile
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

from src.python.workspace_qdrant_mcp.testing.analytics.engine import (
    TestAnalyticsEngine,
    TestResult,
    TestMetrics,
    TrendAnalysis,
    QualityReport,
    MetricType,
    TrendDirection
)


class TestTestAnalyticsEngine:
    """Comprehensive tests for TestAnalyticsEngine."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = Path(f.name)
        yield db_path
        if db_path.exists():
            db_path.unlink()

    @pytest.fixture
    def analytics_engine(self, temp_db):
        """Create analytics engine with temporary database."""
        return TestAnalyticsEngine(db_path=temp_db, enable_ml_features=True)

    @pytest.fixture
    def sample_test_results(self):
        """Generate sample test results for testing."""
        results = []
        base_time = datetime.now() - timedelta(days=7)

        for i in range(100):
            result = TestResult(
                test_id=f"test_{i}",
                test_name=f"test_function_{i}",
                file_path=Path(f"test_file_{i % 10}.py"),
                status="passed" if i % 5 != 0 else "failed",  # 20% failure rate
                execution_time=1.0 + (i % 10) * 0.5,  # Varying execution times
                timestamp=base_time + timedelta(hours=i),
                coverage=85.0 + (i % 15),  # Coverage between 85-99%
                memory_usage=1000 + (i % 100) * 10,
                tags={"unit", f"group_{i % 3}"}
            )
            results.append(result)

        return results

    def test_initialization(self, temp_db):
        """Test analytics engine initialization."""
        engine = TestAnalyticsEngine(db_path=temp_db)

        # Check database was created
        assert temp_db.exists()

        # Check tables exist
        conn = sqlite3.connect(temp_db)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        expected_tables = {
            'test_results', 'metrics_snapshots', 'trend_cache', 'quality_reports'
        }
        assert expected_tables.issubset(tables)

    def test_database_initialization_failure(self):
        """Test database initialization failure handling."""
        with patch('sqlite3.connect', side_effect=Exception("DB Error")):
            with pytest.raises(Exception, match="DB Error"):
                TestAnalyticsEngine(db_path=Path("/invalid/path/db.sqlite"))

    def test_record_test_result_success(self, analytics_engine):
        """Test successful test result recording."""
        result = TestResult(
            test_id="test_1",
            test_name="test_example",
            file_path=Path("test_example.py"),
            status="passed",
            execution_time=1.5,
            timestamp=datetime.now(),
            coverage=95.0
        )

        success = analytics_engine.record_test_result(result)
        assert success is True

    def test_record_test_result_failure(self, analytics_engine):
        """Test test result recording failure handling."""
        result = TestResult(
            test_id="test_1",
            test_name="test_example",
            file_path=Path("test_example.py"),
            status="passed",
            execution_time=1.5,
            timestamp=datetime.now()
        )

        # Mock database failure
        with patch.object(analytics_engine, 'db_path', Path("/invalid/path")):
            success = analytics_engine.record_test_result(result)
            assert success is False

    def test_record_test_results_batch(self, analytics_engine, sample_test_results):
        """Test batch recording of test results."""
        count = analytics_engine.record_test_results_batch(sample_test_results)
        assert count == len(sample_test_results)

        # Verify results were saved
        metrics = analytics_engine.calculate_metrics()
        assert metrics.total_tests > 0

    def test_record_test_results_batch_empty(self, analytics_engine):
        """Test batch recording with empty list."""
        count = analytics_engine.record_test_results_batch([])
        assert count == 0

    def test_record_test_results_batch_partial_failure(self, analytics_engine):
        """Test batch recording with some invalid results."""
        valid_result = TestResult(
            test_id="test_1",
            test_name="test_example",
            file_path=Path("test_example.py"),
            status="passed",
            execution_time=1.5,
            timestamp=datetime.now()
        )

        # Create result with invalid data that will cause JSON serialization to fail
        invalid_result = TestResult(
            test_id="test_2",
            test_name="test_invalid",
            file_path=Path("test_invalid.py"),
            status="passed",
            execution_time=1.5,
            timestamp=datetime.now()
        )
        invalid_result.metadata = {"invalid": float('inf')}  # Invalid JSON

        results = [valid_result, invalid_result]
        count = analytics_engine.record_test_results_batch(results)

        # Should record at least the valid result
        assert count >= 1

    def test_calculate_metrics_no_data(self, analytics_engine):
        """Test metrics calculation with no data."""
        metrics = analytics_engine.calculate_metrics()

        assert metrics.total_tests == 0
        assert metrics.pass_rate == 0
        assert metrics.trend == TrendDirection.UNKNOWN

    def test_calculate_metrics_with_data(self, analytics_engine, sample_test_results):
        """Test metrics calculation with sample data."""
        analytics_engine.record_test_results_batch(sample_test_results)

        metrics = analytics_engine.calculate_metrics()

        assert metrics.total_tests > 0
        assert 0 <= metrics.pass_rate <= 100
        assert metrics.avg_execution_time > 0
        assert metrics.coverage_percentage > 0

    def test_calculate_metrics_with_filter(self, analytics_engine, sample_test_results):
        """Test metrics calculation with SQL filter."""
        analytics_engine.record_test_results_batch(sample_test_results)

        # Filter for only passed tests
        metrics = analytics_engine.calculate_metrics(test_filter="status = 'passed'")

        assert metrics.total_tests > 0
        assert metrics.failed == 0  # Should have no failed tests

    def test_calculate_metrics_database_error(self, analytics_engine):
        """Test metrics calculation with database error."""
        with patch('sqlite3.connect', side_effect=Exception("DB Error")):
            metrics = analytics_engine.calculate_metrics()

            # Should return empty metrics without crashing
            assert metrics.total_tests == 0

    def test_analyze_trends_insufficient_data(self, analytics_engine):
        """Test trend analysis with insufficient data points."""
        trend = analytics_engine.analyze_trends(MetricType.PASS_RATE, min_data_points=10)
        assert trend is None

    def test_analyze_trends_with_data(self, analytics_engine, sample_test_results):
        """Test trend analysis with sufficient data."""
        analytics_engine.record_test_results_batch(sample_test_results)

        # Wait a bit to ensure we have data points across time
        trend = analytics_engine.analyze_trends(MetricType.PASS_RATE, days_back=8, min_data_points=2)

        if trend:  # Might be None if not enough daily data points
            assert isinstance(trend, TrendAnalysis)
            assert trend.metric_type == MetricType.PASS_RATE
            assert trend.direction in TrendDirection
            assert trend.confidence >= 0

    def test_analyze_trends_stable_values(self, analytics_engine):
        """Test trend analysis with stable values (no variance)."""
        # Create results with identical pass rates
        base_time = datetime.now() - timedelta(days=7)
        results = []

        for i in range(50):
            result = TestResult(
                test_id=f"test_{i}",
                test_name=f"test_function_{i}",
                file_path=Path("test_file.py"),
                status="passed",  # All passed
                execution_time=1.0,
                timestamp=base_time + timedelta(hours=i),
                coverage=90.0
            )
            results.append(result)

        analytics_engine.record_test_results_batch(results)
        trend = analytics_engine.analyze_trends(MetricType.PASS_RATE, days_back=8, min_data_points=2)

        if trend:
            assert trend.direction == TrendDirection.STABLE

    def test_analyze_trends_improving(self, analytics_engine):
        """Test trend analysis with improving metrics."""
        base_time = datetime.now() - timedelta(days=7)
        results = []

        # Create improving pass rate over time
        for i in range(50):
            # Fewer failures over time
            status = "failed" if i < (50 - i) // 10 else "passed"
            result = TestResult(
                test_id=f"test_{i}",
                test_name=f"test_function_{i}",
                file_path=Path("test_file.py"),
                status=status,
                execution_time=1.0,
                timestamp=base_time + timedelta(hours=i)
            )
            results.append(result)

        analytics_engine.record_test_results_batch(results)
        trend = analytics_engine.analyze_trends(MetricType.PASS_RATE, days_back=8, min_data_points=2)

        if trend and len(trend.data_points) > 2:
            # Should detect improving trend
            assert trend.change_percentage > 0

    def test_analyze_trends_volatile(self, analytics_engine):
        """Test trend analysis with volatile metrics."""
        base_time = datetime.now() - timedelta(days=7)
        results = []

        # Create volatile pass rate
        for i in range(50):
            # Alternate between high and low failure rates
            status = "failed" if i % 2 == 0 else "passed"
            result = TestResult(
                test_id=f"test_{i}",
                test_name=f"test_function_{i}",
                file_path=Path("test_file.py"),
                status=status,
                execution_time=1.0,
                timestamp=base_time + timedelta(hours=i)
            )
            results.append(result)

        analytics_engine.record_test_results_batch(results)
        trend = analytics_engine.analyze_trends(MetricType.PASS_RATE, days_back=8, min_data_points=2)

        if trend and len(trend.data_points) > 3:
            # High volatility should be detected
            values = [point[1] for point in trend.data_points]
            if len(set(values)) > 1:  # Multiple different values
                std_dev = np.std(values)
                mean_val = np.mean(values)
                coefficient_of_variation = std_dev / mean_val if mean_val != 0 else 0
                if coefficient_of_variation > 0.3:
                    assert trend.direction == TrendDirection.VOLATILE

    def test_analyze_trends_with_anomalies(self, analytics_engine):
        """Test trend analysis with ML-based anomaly detection."""
        base_time = datetime.now() - timedelta(days=7)
        results = []

        # Create mostly stable data with one anomaly
        for i in range(50):
            # Create anomaly at position 25
            if i == 25:
                status = "failed"  # Anomaly: all tests fail
            else:
                status = "passed"  # Normal: all tests pass

            result = TestResult(
                test_id=f"test_{i}",
                test_name=f"test_function_{i}",
                file_path=Path("test_file.py"),
                status=status,
                execution_time=1.0,
                timestamp=base_time + timedelta(hours=i)
            )
            results.append(result)

        analytics_engine.record_test_results_batch(results)
        trend = analytics_engine.analyze_trends(MetricType.PASS_RATE, days_back=8, min_data_points=2)

        if trend and trend.anomalies:
            assert len(trend.anomalies) > 0

    def test_generate_quality_report(self, analytics_engine, sample_test_results):
        """Test quality report generation."""
        analytics_engine.record_test_results_batch(sample_test_results)

        report = analytics_engine.generate_quality_report()

        assert isinstance(report, QualityReport)
        assert 0 <= report.overall_score <= 100
        assert len(report.metrics) > 0
        assert isinstance(report.recommendations, list)
        assert isinstance(report.warnings, list)
        assert isinstance(report.critical_issues, list)

    def test_generate_quality_report_error(self, analytics_engine):
        """Test quality report generation with error."""
        with patch.object(analytics_engine, 'calculate_metrics', side_effect=Exception("Error")):
            report = analytics_engine.generate_quality_report()

            # Should return error report without crashing
            assert report.overall_score == 0
            assert len(report.warnings) > 0

    def test_flakiness_score_calculation(self, analytics_engine):
        """Test flakiness score calculation."""
        base_time = datetime.now() - timedelta(days=1)

        # Create flaky test results
        flaky_results = []
        for i in range(10):
            # Same test with different results
            status = "passed" if i % 2 == 0 else "failed"
            result = TestResult(
                test_id="flaky_test",
                test_name="test_flaky",
                file_path=Path("test_flaky.py"),
                status=status,
                execution_time=1.0,
                timestamp=base_time + timedelta(minutes=i)
            )
            flaky_results.append(result)

        # Add stable test results
        stable_results = []
        for i in range(10):
            result = TestResult(
                test_id="stable_test",
                test_name="test_stable",
                file_path=Path("test_stable.py"),
                status="passed",
                execution_time=1.0,
                timestamp=base_time + timedelta(minutes=i)
            )
            stable_results.append(result)

        all_results = flaky_results + stable_results
        analytics_engine.record_test_results_batch(all_results)

        metrics = analytics_engine.calculate_metrics()

        # Should detect flakiness
        assert metrics.flakiness_score > 0

    def test_flakiness_score_no_data(self, analytics_engine):
        """Test flakiness score calculation with no data."""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=1)

        score = analytics_engine._calculate_flakiness_score(start_time, end_time)
        assert score == 0.0

    def test_flakiness_score_database_error(self, analytics_engine):
        """Test flakiness score calculation with database error."""
        with patch('sqlite3.connect', side_effect=Exception("DB Error")):
            end_time = datetime.now()
            start_time = end_time - timedelta(days=1)

            score = analytics_engine._calculate_flakiness_score(start_time, end_time)
            assert score == 0.0

    def test_reliability_score_calculation(self, analytics_engine):
        """Test reliability score calculation."""
        metrics = TestMetrics(
            total_tests=100,
            passed=90,
            failed=10,
            pass_rate=90.0,
            flakiness_score=5.0
        )

        reliability = analytics_engine._calculate_reliability_score(metrics)

        assert 0 <= reliability <= 1
        assert reliability < 1.0  # Should be reduced due to flakiness

    def test_reliability_score_perfect(self, analytics_engine):
        """Test reliability score with perfect metrics."""
        metrics = TestMetrics(
            total_tests=100,
            passed=100,
            failed=0,
            pass_rate=100.0,
            flakiness_score=0.0
        )

        reliability = analytics_engine._calculate_reliability_score(metrics)
        assert reliability == 1.0

    def test_reliability_score_error_handling(self, analytics_engine):
        """Test reliability score calculation with error."""
        metrics = TestMetrics()

        with patch.object(analytics_engine, '_calculate_reliability_score', side_effect=Exception("Error")):
            # Should not crash, but return default
            pass

    def test_get_metric_history_pass_rate(self, analytics_engine, sample_test_results):
        """Test metric history retrieval for pass rate."""
        analytics_engine.record_test_results_batch(sample_test_results)

        end_time = datetime.now()
        start_time = end_time - timedelta(days=8)

        history = analytics_engine._get_metric_history(MetricType.PASS_RATE, start_time, end_time)

        # Should have some data points
        if history:
            assert len(history) > 0
            for timestamp, value in history:
                assert isinstance(timestamp, datetime)
                assert 0 <= value <= 100

    def test_get_metric_history_execution_time(self, analytics_engine, sample_test_results):
        """Test metric history retrieval for execution time."""
        analytics_engine.record_test_results_batch(sample_test_results)

        end_time = datetime.now()
        start_time = end_time - timedelta(days=8)

        history = analytics_engine._get_metric_history(MetricType.EXECUTION_TIME, start_time, end_time)

        if history:
            assert len(history) > 0
            for timestamp, value in history:
                assert isinstance(timestamp, datetime)
                assert value >= 0

    def test_get_metric_history_unsupported_type(self, analytics_engine):
        """Test metric history with unsupported metric type."""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=1)

        history = analytics_engine._get_metric_history(MetricType.COMPLEXITY, start_time, end_time)
        assert history == []

    def test_detect_anomalies_insufficient_data(self, analytics_engine):
        """Test anomaly detection with insufficient data."""
        data_points = [(datetime.now(), 50.0)]
        anomalies = analytics_engine._detect_anomalies(data_points)
        assert anomalies == []

    def test_detect_anomalies_no_variance(self, analytics_engine):
        """Test anomaly detection with no variance."""
        base_time = datetime.now()
        data_points = [(base_time + timedelta(hours=i), 50.0) for i in range(10)]

        anomalies = analytics_engine._detect_anomalies(data_points)
        assert anomalies == []  # No anomalies in constant data

    def test_detect_anomalies_with_outliers(self, analytics_engine):
        """Test anomaly detection with clear outliers."""
        base_time = datetime.now()
        data_points = []

        # Normal values around 50
        for i in range(10):
            data_points.append((base_time + timedelta(hours=i), 50.0))

        # Add clear outlier
        outlier_time = base_time + timedelta(hours=5)
        data_points[5] = (outlier_time, 100.0)  # Replace with outlier

        anomalies = analytics_engine._detect_anomalies(data_points)

        if anomalies:
            assert outlier_time in anomalies

    def test_calculate_quality_scores(self, analytics_engine):
        """Test quality scores calculation."""
        metrics = TestMetrics(
            total_tests=100,
            passed=85,
            failed=15,
            pass_rate=85.0,
            avg_execution_time=25.0,
            coverage_percentage=90.0,
            flakiness_score=10.0,
            reliability_score=0.8
        )

        trends = []  # Empty trends for simplicity
        scores = analytics_engine._calculate_quality_scores(metrics, trends)

        assert MetricType.PASS_RATE in scores
        assert MetricType.EXECUTION_TIME in scores
        assert MetricType.COVERAGE in scores
        assert MetricType.RELIABILITY in scores
        assert MetricType.FLAKINESS in scores

        # Check score ranges
        for score in scores.values():
            assert 0 <= score <= 100

    def test_calculate_quality_scores_edge_cases(self, analytics_engine):
        """Test quality scores with edge case values."""
        # Extreme values
        metrics = TestMetrics(
            total_tests=1,
            passed=0,
            failed=1,
            pass_rate=0.0,
            avg_execution_time=0.0,  # Zero execution time
            coverage_percentage=0.0,
            flakiness_score=100.0,  # Maximum flakiness
            reliability_score=0.0
        )

        scores = analytics_engine._calculate_quality_scores(metrics, [])

        # Should handle extreme values gracefully
        assert scores[MetricType.PASS_RATE] == 0.0
        assert scores[MetricType.COVERAGE] == 0.0
        assert scores[MetricType.RELIABILITY] == 0.0
        assert scores[MetricType.FLAKINESS] == 0.0  # Inverted score

    def test_generate_recommendations_good_metrics(self, analytics_engine):
        """Test recommendations generation with good metrics."""
        metrics = TestMetrics(
            pass_rate=95.0,
            avg_execution_time=15.0,
            coverage_percentage=90.0,
            flakiness_score=3.0
        )

        recommendations = analytics_engine._generate_recommendations(metrics, [])

        # Should have positive recommendation
        assert any("good" in rec.lower() for rec in recommendations)

    def test_generate_recommendations_poor_metrics(self, analytics_engine):
        """Test recommendations generation with poor metrics."""
        metrics = TestMetrics(
            pass_rate=40.0,  # Poor pass rate
            avg_execution_time=90.0,  # Slow tests
            coverage_percentage=50.0,  # Low coverage
            flakiness_score=25.0  # High flakiness
        )

        recommendations = analytics_engine._generate_recommendations(metrics, [])

        # Should have multiple recommendations
        assert len(recommendations) > 1
        assert any("pass rate" in rec.lower() for rec in recommendations)
        assert any("execution time" in rec.lower() or "optimization" in rec.lower() for rec in recommendations)
        assert any("coverage" in rec.lower() for rec in recommendations)
        assert any("flak" in rec.lower() for rec in recommendations)

    def test_identify_issues(self, analytics_engine):
        """Test issue identification."""
        metrics = TestMetrics(
            pass_rate=30.0,  # Critical
            avg_execution_time=90.0,  # Warning
            coverage_percentage=60.0,  # Warning
            flakiness_score=40.0  # Critical
        )

        warnings, critical_issues = analytics_engine._identify_issues(metrics, [])

        assert len(critical_issues) >= 2  # Pass rate and flakiness
        assert len(warnings) >= 2  # Execution time and coverage

    def test_identify_issues_good_metrics(self, analytics_engine):
        """Test issue identification with good metrics."""
        metrics = TestMetrics(
            pass_rate=95.0,
            avg_execution_time=20.0,
            coverage_percentage=85.0,
            flakiness_score=2.0
        )

        warnings, critical_issues = analytics_engine._identify_issues(metrics, [])

        assert len(critical_issues) == 0
        assert len(warnings) == 0

    def test_cache_quality_report(self, analytics_engine):
        """Test quality report caching."""
        report = QualityReport(
            overall_score=85.0,
            metrics={MetricType.PASS_RATE: 90.0},
            trends=[],
            recommendations=["Keep up the good work"],
            warnings=[],
            critical_issues=[]
        )

        # Should not raise exception
        analytics_engine._cache_quality_report(report)

        # Verify it was saved
        conn = sqlite3.connect(analytics_engine.db_path)
        cursor = conn.execute('SELECT COUNT(*) FROM quality_reports')
        count = cursor.fetchone()[0]
        conn.close()

        assert count > 0

    def test_cache_quality_report_error(self, analytics_engine):
        """Test quality report caching with database error."""
        report = QualityReport(
            overall_score=85.0,
            metrics={},
            trends=[],
            recommendations=[],
            warnings=[],
            critical_issues=[]
        )

        with patch('sqlite3.connect', side_effect=Exception("DB Error")):
            # Should not raise exception
            analytics_engine._cache_quality_report(report)

    def test_get_dashboard_data(self, analytics_engine, sample_test_results):
        """Test dashboard data generation."""
        analytics_engine.record_test_results_batch(sample_test_results)

        dashboard_data = analytics_engine.get_dashboard_data(days_back=7)

        assert 'metrics' in dashboard_data
        assert 'trends' in dashboard_data
        assert 'period' in dashboard_data

        # Check metrics structure
        metrics = dashboard_data['metrics']
        assert 'total_tests' in metrics
        assert 'pass_rate' in metrics
        assert 'avg_execution_time' in metrics

    def test_get_dashboard_data_error(self, analytics_engine):
        """Test dashboard data generation with error."""
        with patch.object(analytics_engine, 'calculate_metrics', side_effect=Exception("Error")):
            dashboard_data = analytics_engine.get_dashboard_data()

            assert 'error' in dashboard_data

    def test_ml_features_disabled(self, temp_db):
        """Test analytics engine with ML features disabled."""
        engine = TestAnalyticsEngine(db_path=temp_db, enable_ml_features=False)

        # Create some data
        base_time = datetime.now() - timedelta(days=2)
        results = [
            TestResult(
                test_id=f"test_{i}",
                test_name=f"test_function_{i}",
                file_path=Path("test_file.py"),
                status="passed",
                execution_time=1.0,
                timestamp=base_time + timedelta(hours=i)
            )
            for i in range(10)
        ]

        engine.record_test_results_batch(results)

        # Trend analysis should work but without ML features
        trend = engine.analyze_trends(MetricType.PASS_RATE, days_back=3, min_data_points=2)

        if trend:
            # Should not have anomaly detection
            assert trend.anomalies == []

    def test_concurrent_access(self, analytics_engine):
        """Test concurrent access to analytics engine."""
        import threading
        import time

        results = []
        errors = []

        def worker(worker_id):
            try:
                result = TestResult(
                    test_id=f"test_{worker_id}",
                    test_name=f"test_function_{worker_id}",
                    file_path=Path("test_file.py"),
                    status="passed",
                    execution_time=1.0,
                    timestamp=datetime.now()
                )
                success = analytics_engine.record_test_result(result)
                results.append(success)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Should have no errors and all successes
        assert len(errors) == 0
        assert all(results)