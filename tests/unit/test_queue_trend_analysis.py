"""Unit tests for queue trend analysis module.

Tests cover:
- Data storage and retrieval
- Linear regression calculations
- Trend direction detection
- Forecasting
- Anomaly detection
- Period comparison
- Edge cases and error handling
"""

import asyncio
import json
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.python.common.core.queue_trend_analysis import (
    Anomaly,
    HistoricalTrendAnalyzer,
    PeriodComparison,
    TrendAnalysis,
    TrendDataPoint,
    TrendDirection,
)


@pytest.fixture
async def analyzer():
    """Create a trend analyzer with temporary database."""
    # Use temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    analyzer = HistoricalTrendAnalyzer(db_path=db_path)
    await analyzer.initialize()

    yield analyzer

    # Cleanup
    await analyzer.close()
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def sample_data_points() -> list[TrendDataPoint]:
    """Create sample data points for testing."""
    base_time = datetime.now(timezone.utc) - timedelta(hours=24)
    data_points = []

    # Create 24 hours of hourly data with linear trend
    for i in range(24):
        timestamp = base_time + timedelta(hours=i)
        value = 100.0 + (i * 5.0)  # Linear increase: slope = 5.0 per hour
        data_points.append(TrendDataPoint(
            timestamp=timestamp,
            metric_name="queue_size",
            value=value,
            metadata={"hour": i}
        ))

    return data_points


class TestDataStorage:
    """Tests for metric data storage and retrieval."""

    @pytest.mark.asyncio
    async def test_store_metric_point(self, analyzer):
        """Test storing a single metric point."""
        await analyzer.store_metric_point(
            metric_name="queue_size",
            value=150.0,
            metadata={"collection": "test"}
        )

        # Verify it was stored
        data = await analyzer.get_historical_data("queue_size", hours=1)
        assert len(data) == 1
        assert data[0].metric_name == "queue_size"
        assert data[0].value == 150.0
        assert data[0].metadata["collection"] == "test"

    @pytest.mark.asyncio
    async def test_store_multiple_points(self, analyzer):
        """Test storing multiple metric points."""
        for i in range(10):
            await analyzer.store_metric_point(
                metric_name="processing_rate",
                value=float(i * 10),
                metadata={"iteration": i}
            )

        data = await analyzer.get_historical_data("processing_rate", hours=1)
        assert len(data) == 10
        assert [dp.value for dp in data] == [float(i * 10) for i in range(10)]

    @pytest.mark.asyncio
    async def test_get_historical_data_time_window(self, analyzer):
        """Test retrieving data within a specific time window."""
        base_time = datetime.now(timezone.utc)

        # Store data at different times
        for hours_ago in [1, 3, 6, 12, 24, 48]:
            timestamp = base_time - timedelta(hours=hours_ago)
            # Store with specific timestamp (requires direct DB insert for test)
            analyzer._conn.execute("""
                INSERT INTO metric_history (timestamp, metric_name, value, metadata)
                VALUES (?, ?, ?, ?)
            """, (timestamp.isoformat(), "latency", float(hours_ago), None))
        analyzer._conn.commit()

        # Get last 12 hours
        data_12h = await analyzer.get_historical_data("latency", hours=12)
        assert len(data_12h) == 4  # 1, 3, 6, 12 hours ago

        # Get last 24 hours
        data_24h = await analyzer.get_historical_data("latency", hours=24)
        assert len(data_24h) == 5  # 1, 3, 6, 12, 24 hours ago

    @pytest.mark.asyncio
    async def test_data_point_ordering(self, analyzer):
        """Test that data points are returned in chronological order."""
        # Store in random order
        for value in [30, 10, 20]:
            await asyncio.sleep(0.01)  # Small delay to ensure different timestamps
            await analyzer.store_metric_point("queue_size", float(value))

        data = await analyzer.get_historical_data("queue_size", hours=1)
        # Should be ordered by timestamp ascending
        assert len(data) == 3
        # Values should be in insertion order due to chronological timestamps
        assert data[0].value == 30.0
        assert data[1].value == 10.0
        assert data[2].value == 20.0


class TestLinearRegression:
    """Tests for linear regression calculations."""

    def test_calculate_linear_regression(self, analyzer, sample_data_points):
        """Test linear regression calculation with known slope."""
        # sample_data_points has slope of 5.0 per hour
        slope, intercept, r_squared = analyzer._calculate_linear_regression(sample_data_points)

        assert abs(slope - 5.0) < 0.01  # Should be very close to 5.0
        assert r_squared > 0.99  # Should have excellent fit for linear data

    def test_linear_regression_perfect_fit(self, analyzer):
        """Test linear regression with perfect linear data."""
        base_time = datetime.now(timezone.utc)
        data_points = []

        # Perfect linear relationship: y = 2x + 10
        for i in range(10):
            timestamp = base_time + timedelta(hours=i)
            value = 2.0 * i + 10.0
            data_points.append(TrendDataPoint(
                timestamp=timestamp,
                metric_name="test",
                value=value
            ))

        slope, intercept, r_squared = analyzer._calculate_linear_regression(data_points)

        assert abs(slope - 2.0) < 0.001
        assert abs(intercept - 10.0) < 0.001
        assert r_squared > 0.999  # Perfect fit

    def test_linear_regression_insufficient_data(self, analyzer):
        """Test linear regression with insufficient data points."""
        base_time = datetime.now(timezone.utc)
        data_points = [TrendDataPoint(
            timestamp=base_time,
            metric_name="test",
            value=100.0
        )]

        with pytest.raises(ValueError, match="at least 2 data points"):
            analyzer._calculate_linear_regression(data_points)

    def test_linear_regression_constant_values(self, analyzer):
        """Test linear regression with all same values."""
        base_time = datetime.now(timezone.utc)
        data_points = []

        # All values are 50.0
        for i in range(10):
            timestamp = base_time + timedelta(hours=i)
            data_points.append(TrendDataPoint(
                timestamp=timestamp,
                metric_name="test",
                value=50.0
            ))

        slope, intercept, r_squared = analyzer._calculate_linear_regression(data_points)

        assert abs(slope) < 0.001  # Slope should be near zero
        assert abs(intercept - 50.0) < 0.001  # Intercept should be the constant value


class TestTrendDirection:
    """Tests for trend direction detection."""

    def test_trend_direction_increasing(self, analyzer):
        """Test detection of increasing trend."""
        values = [10.0, 15.0, 20.0, 25.0, 30.0]
        slope = 5.0  # Positive slope > threshold

        direction = analyzer._determine_trend_direction(slope, values)
        assert direction == TrendDirection.INCREASING

    def test_trend_direction_decreasing(self, analyzer):
        """Test detection of decreasing trend."""
        values = [30.0, 25.0, 20.0, 15.0, 10.0]
        slope = -5.0  # Negative slope < -threshold

        direction = analyzer._determine_trend_direction(slope, values)
        assert direction == TrendDirection.DECREASING

    def test_trend_direction_stable(self, analyzer):
        """Test detection of stable trend."""
        values = [100.0, 101.0, 100.5, 100.2, 99.8]
        slope = 0.01  # Very small slope

        direction = analyzer._determine_trend_direction(slope, values)
        assert direction == TrendDirection.STABLE

    def test_trend_direction_volatile(self, analyzer):
        """Test detection of volatile trend."""
        # High variance relative to mean
        values = [10.0, 50.0, 5.0, 45.0, 15.0, 40.0]
        slope = 0.5  # Some slope but high volatility

        direction = analyzer._determine_trend_direction(slope, values)
        assert direction == TrendDirection.VOLATILE

    def test_trend_direction_edge_case_single_value(self, analyzer):
        """Test trend direction with single value."""
        values = [100.0]
        slope = 0.0

        direction = analyzer._determine_trend_direction(slope, values)
        assert direction == TrendDirection.STABLE


class TestTrendAnalysis:
    """Tests for complete trend analysis."""

    @pytest.mark.asyncio
    async def test_get_trend_analysis_increasing(self, analyzer, sample_data_points):
        """Test trend analysis for increasing data."""
        # Store sample data
        for dp in sample_data_points:
            analyzer._conn.execute("""
                INSERT INTO metric_history (timestamp, metric_name, value, metadata)
                VALUES (?, ?, ?, ?)
            """, (dp.timestamp.isoformat(), dp.metric_name, dp.value,
                  json.dumps(dp.metadata) if dp.metadata else None))
        analyzer._conn.commit()

        analysis = await analyzer.get_trend_analysis("queue_size", window_hours=24)

        assert analysis.metric_name == "queue_size"
        assert analysis.trend_direction == TrendDirection.INCREASING
        assert analysis.slope > 4.5  # Should be close to 5.0
        assert analysis.slope < 5.5
        assert analysis.confidence > 0.99  # High R²
        assert analysis.data_points_count == 24
        assert len(analysis.forecast) > 0

    @pytest.mark.asyncio
    async def test_get_trend_analysis_insufficient_data(self, analyzer):
        """Test trend analysis with insufficient data."""
        await analyzer.store_metric_point("error_rate", 5.0)

        with pytest.raises(ValueError, match="Insufficient data"):
            await analyzer.get_trend_analysis("error_rate", window_hours=1)

    @pytest.mark.asyncio
    async def test_trend_analysis_forecast_values(self, analyzer, sample_data_points):
        """Test that forecast values are generated."""
        # Store sample data
        for dp in sample_data_points:
            analyzer._conn.execute("""
                INSERT INTO metric_history (timestamp, metric_name, value, metadata)
                VALUES (?, ?, ?, ?)
            """, (dp.timestamp.isoformat(), dp.metric_name, dp.value,
                  json.dumps(dp.metadata) if dp.metadata else None))
        analyzer._conn.commit()

        analysis = await analyzer.get_trend_analysis("queue_size")

        # Should have forecasts for 1, 6, 12, 24 hours
        assert "1h" in analysis.forecast
        assert "6h" in analysis.forecast
        assert "12h" in analysis.forecast
        assert "24h" in analysis.forecast

        # With positive slope, forecasts should increase
        assert analysis.forecast["1h"] < analysis.forecast["6h"]
        assert analysis.forecast["6h"] < analysis.forecast["12h"]

        # Forecasts should not be negative
        assert all(v >= 0 for v in analysis.forecast.values())


class TestForecasting:
    """Tests for metric forecasting."""

    @pytest.mark.asyncio
    async def test_forecast_metric_increasing_trend(self, analyzer, sample_data_points):
        """Test forecasting with increasing trend."""
        # Store sample data (slope = 5.0 per hour)
        for dp in sample_data_points:
            analyzer._conn.execute("""
                INSERT INTO metric_history (timestamp, metric_name, value, metadata)
                VALUES (?, ?, ?, ?)
            """, (dp.timestamp.isoformat(), dp.metric_name, dp.value,
                  json.dumps(dp.metadata) if dp.metadata else None))
        analyzer._conn.commit()

        # Forecast 1 hour ahead
        forecast = await analyzer.forecast_metric("queue_size", hours_ahead=1.0)

        # Last value is 100 + (23 * 5) = 215, so 1 hour ahead ≈ 220
        assert forecast > 215.0
        assert forecast < 225.0

    @pytest.mark.asyncio
    async def test_forecast_metric_decreasing_trend(self, analyzer):
        """Test forecasting with decreasing trend."""
        base_time = datetime.now(timezone.utc) - timedelta(hours=10)

        # Create decreasing trend
        for i in range(10):
            timestamp = base_time + timedelta(hours=i)
            value = 100.0 - (i * 5.0)  # Decreasing
            analyzer._conn.execute("""
                INSERT INTO metric_history (timestamp, metric_name, value, metadata)
                VALUES (?, ?, ?, ?)
            """, (timestamp.isoformat(), "queue_size", value, None))
        analyzer._conn.commit()

        forecast = await analyzer.forecast_metric("queue_size", hours_ahead=1.0)

        # Should forecast lower value but not negative
        assert forecast >= 0.0
        assert forecast < 60.0  # Last value was 55, forecast should be around 50

    @pytest.mark.asyncio
    async def test_forecast_no_negative_values(self, analyzer):
        """Test that forecast never returns negative values."""
        base_time = datetime.now(timezone.utc) - timedelta(hours=5)

        # Create sharply decreasing trend that would go negative
        for i in range(5):
            timestamp = base_time + timedelta(hours=i)
            value = 20.0 - (i * 10.0)  # Will go negative
            analyzer._conn.execute("""
                INSERT INTO metric_history (timestamp, metric_name, value, metadata)
                VALUES (?, ?, ?, ?)
            """, (timestamp.isoformat(), "test", value, None))
        analyzer._conn.commit()

        forecast = await analyzer.forecast_metric("test", hours_ahead=5.0)

        # Should be clamped to 0.0
        assert forecast == 0.0


class TestAnomalyDetection:
    """Tests for anomaly detection."""

    @pytest.mark.asyncio
    async def test_detect_anomalies_with_spike(self, analyzer):
        """Test anomaly detection with a clear spike."""
        base_time = datetime.now(timezone.utc) - timedelta(hours=10)

        # Normal values around 100, with one spike
        for i in range(10):
            timestamp = base_time + timedelta(hours=i)
            if i == 5:
                value = 500.0  # Anomaly spike
            else:
                value = 100.0 + (i % 3)  # Normal variation
            analyzer._conn.execute("""
                INSERT INTO metric_history (timestamp, metric_name, value, metadata)
                VALUES (?, ?, ?, ?)
            """, (timestamp.isoformat(), "queue_size", value, None))
        analyzer._conn.commit()

        anomalies = await analyzer.detect_anomalies("queue_size", sensitivity=3.0)

        assert len(anomalies) == 1
        assert anomalies[0].value == 500.0
        assert abs(anomalies[0].z_score) > 3.0
        assert anomalies[0].severity in ["low", "medium", "high", "critical"]

    @pytest.mark.asyncio
    async def test_detect_anomalies_with_drop(self, analyzer):
        """Test anomaly detection with a sudden drop."""
        base_time = datetime.now(timezone.utc) - timedelta(hours=10)

        # Normal values around 100, with one drop
        for i in range(10):
            timestamp = base_time + timedelta(hours=i)
            if i == 5:
                value = 10.0  # Anomaly drop
            else:
                value = 100.0 + (i % 3)
            analyzer._conn.execute("""
                INSERT INTO metric_history (timestamp, metric_name, value, metadata)
                VALUES (?, ?, ?, ?)
            """, (timestamp.isoformat(), "processing_rate", value, None))
        analyzer._conn.commit()

        anomalies = await analyzer.detect_anomalies("processing_rate", sensitivity=3.0)

        assert len(anomalies) == 1
        assert anomalies[0].value == 10.0
        assert anomalies[0].z_score < -3.0  # Negative z-score for below mean

    @pytest.mark.asyncio
    async def test_anomaly_sensitivity(self, analyzer):
        """Test that sensitivity affects anomaly detection."""
        base_time = datetime.now(timezone.utc) - timedelta(hours=20)

        # Create data with moderate outliers
        for i in range(20):
            timestamp = base_time + timedelta(hours=i)
            if i in [5, 15]:
                value = 150.0  # Moderate outliers
            else:
                value = 100.0 + (i % 5)
            analyzer._conn.execute("""
                INSERT INTO metric_history (timestamp, metric_name, value, metadata)
                VALUES (?, ?, ?, ?)
            """, (timestamp.isoformat(), "latency", value, None))
        analyzer._conn.commit()

        # High sensitivity (low threshold) detects more anomalies
        anomalies_sensitive = await analyzer.detect_anomalies("latency", sensitivity=2.0)

        # Low sensitivity (high threshold) detects fewer anomalies
        anomalies_conservative = await analyzer.detect_anomalies("latency", sensitivity=4.0)

        assert len(anomalies_sensitive) >= len(anomalies_conservative)

    @pytest.mark.asyncio
    async def test_no_anomalies_in_stable_data(self, analyzer):
        """Test that stable data has no anomalies."""
        base_time = datetime.now(timezone.utc) - timedelta(hours=10)

        # All values very close to 100
        for i in range(10):
            timestamp = base_time + timedelta(hours=i)
            value = 100.0 + (i % 2) * 0.1  # Minimal variation
            analyzer._conn.execute("""
                INSERT INTO metric_history (timestamp, metric_name, value, metadata)
                VALUES (?, ?, ?, ?)
            """, (timestamp.isoformat(), "success_rate", value, None))
        analyzer._conn.commit()

        anomalies = await analyzer.detect_anomalies("success_rate", sensitivity=3.0)

        assert len(anomalies) == 0


class TestPeriodComparison:
    """Tests for period-over-period comparison."""

    @pytest.mark.asyncio
    async def test_compare_periods_basic(self, analyzer):
        """Test basic period comparison."""
        base_time = datetime.now(timezone.utc)

        # Period 1: Last 24 hours (higher values)
        for i in range(24):
            timestamp = base_time - timedelta(hours=i)
            value = 150.0 + i
            analyzer._conn.execute("""
                INSERT INTO metric_history (timestamp, metric_name, value, metadata)
                VALUES (?, ?, ?, ?)
            """, (timestamp.isoformat(), "queue_size", value, None))

        # Period 2: Previous 24 hours (lower values)
        for i in range(24):
            timestamp = base_time - timedelta(hours=24 + i)
            value = 100.0 + i
            analyzer._conn.execute("""
                INSERT INTO metric_history (timestamp, metric_name, value, metadata)
                VALUES (?, ?, ?, ?)
            """, (timestamp.isoformat(), "queue_size", value, None))

        analyzer._conn.commit()

        comparison = await analyzer.compare_periods("queue_size", 24, 24, 0)

        assert comparison.metric_name == "queue_size"
        assert comparison.period1_stats["mean"] > comparison.period2_stats["mean"]
        assert comparison.change_pct > 0  # Period 1 is higher
        assert comparison.change_absolute > 0

    @pytest.mark.asyncio
    async def test_compare_periods_percentage_change(self, analyzer):
        """Test percentage change calculation."""
        base_time = datetime.now(timezone.utc)

        # Period 1: All 200
        for i in range(10):
            timestamp = base_time - timedelta(hours=i)
            analyzer._conn.execute("""
                INSERT INTO metric_history (timestamp, metric_name, value, metadata)
                VALUES (?, ?, ?, ?)
            """, (timestamp.isoformat(), "test", 200.0, None))

        # Period 2: All 100
        for i in range(10):
            timestamp = base_time - timedelta(hours=10 + i)
            analyzer._conn.execute("""
                INSERT INTO metric_history (timestamp, metric_name, value, metadata)
                VALUES (?, ?, ?, ?)
            """, (timestamp.isoformat(), "test", 100.0, None))

        analyzer._conn.commit()

        comparison = await analyzer.compare_periods("test", 10, 10, 0)

        # 100% increase from 100 to 200
        assert abs(comparison.change_pct - 100.0) < 1.0
        assert abs(comparison.change_absolute - 100.0) < 1.0

    @pytest.mark.asyncio
    async def test_period_comparison_statistics(self, analyzer):
        """Test that period statistics are calculated correctly."""
        base_time = datetime.now(timezone.utc)

        # Period 1: values 1-10
        for i in range(1, 11):
            timestamp = base_time - timedelta(hours=i)
            analyzer._conn.execute("""
                INSERT INTO metric_history (timestamp, metric_name, value, metadata)
                VALUES (?, ?, ?, ?)
            """, (timestamp.isoformat(), "test", float(i), None))

        analyzer._conn.commit()

        # Period 2: values 11-20
        for i in range(11, 21):
            timestamp = base_time - timedelta(hours=i)
            analyzer._conn.execute("""
                INSERT INTO metric_history (timestamp, metric_name, value, metadata)
                VALUES (?, ?, ?, ?)
            """, (timestamp.isoformat(), "test", float(i), None))

        analyzer._conn.commit()

        comparison = await analyzer.compare_periods("test", 10, 10, 0)

        # Period 1 mean: (1+2+...+10)/10 = 5.5
        # Period 2 mean: (11+12+...+20)/10 = 15.5
        assert abs(comparison.period1_stats["mean"] - 5.5) < 0.1
        assert abs(comparison.period2_stats["mean"] - 15.5) < 0.1
        assert comparison.period1_stats["min"] == 1.0
        assert comparison.period1_stats["max"] == 10.0
        assert comparison.period2_stats["min"] == 11.0
        assert comparison.period2_stats["max"] == 20.0


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_empty_data(self, analyzer):
        """Test handling of empty data."""
        data = await analyzer.get_historical_data("nonexistent", hours=24)
        assert len(data) == 0

    @pytest.mark.asyncio
    async def test_all_same_values(self, analyzer):
        """Test trend analysis with all identical values."""
        base_time = datetime.now(timezone.utc) - timedelta(hours=10)

        for i in range(10):
            timestamp = base_time + timedelta(hours=i)
            analyzer._conn.execute("""
                INSERT INTO metric_history (timestamp, metric_name, value, metadata)
                VALUES (?, ?, ?, ?)
            """, (timestamp.isoformat(), "constant", 42.0, None))
        analyzer._conn.commit()

        analysis = await analyzer.get_trend_analysis("constant")

        assert analysis.trend_direction == TrendDirection.STABLE
        assert abs(analysis.slope) < 0.01
        assert abs(analysis.mean - 42.0) < 0.01

    @pytest.mark.asyncio
    async def test_highly_volatile_data(self, analyzer):
        """Test trend analysis with highly volatile data."""
        base_time = datetime.now(timezone.utc) - timedelta(hours=20)

        # Random-looking values with high variance
        values = [10, 90, 20, 80, 30, 70, 40, 60, 50, 100, 5, 95]
        for i, value in enumerate(values):
            timestamp = base_time + timedelta(hours=i)
            analyzer._conn.execute("""
                INSERT INTO metric_history (timestamp, metric_name, value, metadata)
                VALUES (?, ?, ?, ?)
            """, (timestamp.isoformat(), "volatile", float(value), None))
        analyzer._conn.commit()

        analysis = await analyzer.get_trend_analysis("volatile")

        # Should be classified as volatile
        assert analysis.trend_direction == TrendDirection.VOLATILE
        assert analysis.volatility_ratio > analyzer.volatile_coefficient_threshold


class TestDataRetention:
    """Tests for data cleanup and retention."""

    @pytest.mark.asyncio
    async def test_cleanup_old_data(self, analyzer):
        """Test cleanup of old data."""
        # Override retention for testing
        analyzer.retention_days = 1

        base_time = datetime.now(timezone.utc)

        # Store old data (3 days ago)
        old_timestamp = base_time - timedelta(days=3)
        analyzer._conn.execute("""
            INSERT INTO metric_history (timestamp, metric_name, value, metadata)
            VALUES (?, ?, ?, ?)
        """, (old_timestamp.isoformat(), "test", 100.0, None))

        # Store recent data (12 hours ago)
        recent_timestamp = base_time - timedelta(hours=12)
        analyzer._conn.execute("""
            INSERT INTO metric_history (timestamp, metric_name, value, metadata)
            VALUES (?, ?, ?, ?)
        """, (recent_timestamp.isoformat(), "test", 200.0, None))

        analyzer._conn.commit()

        # Run cleanup
        deleted_count = await analyzer._cleanup_old_data()

        # Should have deleted 1 old record
        assert deleted_count == 1

        # Verify recent data still exists
        data = await analyzer.get_historical_data("test", hours=24)
        assert len(data) == 1
        assert data[0].value == 200.0

    @pytest.mark.asyncio
    async def test_export_trends(self, analyzer, sample_data_points):
        """Test exporting trend data to JSON."""
        # Store sample data
        for dp in sample_data_points:
            analyzer._conn.execute("""
                INSERT INTO metric_history (timestamp, metric_name, value, metadata)
                VALUES (?, ?, ?, ?)
            """, (dp.timestamp.isoformat(), dp.metric_name, dp.value,
                  json.dumps(dp.metadata) if dp.metadata else None))
        analyzer._conn.commit()

        # Export trends
        export_json = await analyzer.export_trends(format='json')

        # Parse JSON
        export_data = json.loads(export_json)

        assert "timestamp" in export_data
        assert "metrics" in export_data
        assert "queue_size" in export_data["metrics"]

        # Check queue_size data
        queue_data = export_data["metrics"]["queue_size"]
        assert queue_data["trend_direction"] == "increasing"
        assert queue_data["data_points_count"] == 24
        assert queue_data["slope"] > 0
        assert "forecast" in queue_data

    @pytest.mark.asyncio
    async def test_export_trends_unsupported_format(self, analyzer):
        """Test export with unsupported format raises error."""
        with pytest.raises(ValueError, match="Unsupported export format"):
            await analyzer.export_trends(format='xml')


class TestConfiguration:
    """Tests for configuration loading."""

    @pytest.mark.asyncio
    async def test_analyzer_uses_configuration(self):
        """Test that analyzer respects configuration values."""
        # Create analyzer (will load from config or use defaults)
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        analyzer = HistoricalTrendAnalyzer(db_path=db_path)
        await analyzer.initialize()

        try:
            # Check default configuration values
            assert analyzer.enabled is True
            assert analyzer.retention_days == 30
            assert analyzer.default_window_hours == 24
            assert analyzer.anomaly_sensitivity == 3.0
            assert "queue_size" in analyzer.metrics_to_track
        finally:
            await analyzer.close()
            Path(db_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_metrics_to_track_validation(self, analyzer):
        """Test that only configured metrics can be stored."""
        # Try to store an unconfigured metric
        await analyzer.store_metric_point("unknown_metric", 100.0)

        # Should be ignored (warning logged)
        data = await analyzer.get_historical_data("unknown_metric", hours=1)
        assert len(data) == 0

        # Configured metric should work
        await analyzer.store_metric_point("queue_size", 100.0)
        data = await analyzer.get_historical_data("queue_size", hours=1)
        assert len(data) == 1
