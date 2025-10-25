"""Historical trend analysis for queue metrics.

This module provides comprehensive trend analysis, forecasting, and anomaly
detection for queue performance metrics. It stores historical data points
in SQLite and performs statistical analysis to identify patterns, predict
future behavior, and detect anomalies.

Features:
    - Historical metric storage with configurable retention
    - Linear regression for trend analysis
    - Time-series forecasting
    - Z-score based anomaly detection
    - Period-over-period comparison
    - Automatic data cleanup
    - JSON export of trend analyses

Tracked Metrics:
    - queue_size: Number of items in queue over time
    - processing_rate: Items processed per unit time
    - error_rate: Percentage of failed operations
    - latency: Average queue wait time
    - success_rate: Percentage of successful operations
    - resource_usage_cpu: CPU utilization percentage
    - resource_usage_memory: Memory usage in MB

Example:
    ```python
    from workspace_qdrant_mcp.core.queue_trend_analysis import HistoricalTrendAnalyzer

    # Initialize analyzer
    analyzer = HistoricalTrendAnalyzer(db_path="state.db")
    await analyzer.initialize()

    # Store a metric point
    await analyzer.store_metric_point(
        metric_name="queue_size",
        value=150.0,
        metadata={"collection": "my-project"}
    )

    # Get trend analysis
    analysis = await analyzer.get_trend_analysis("queue_size", window_hours=24)
    print(f"Trend: {analysis.trend_direction}, Slope: {analysis.slope}")

    # Forecast future value
    future_value = await analyzer.forecast_metric("queue_size", hours_ahead=1)

    # Detect anomalies
    anomalies = await analyzer.detect_anomalies("queue_size", sensitivity=3.0)
    ```
"""

import asyncio
import json
import sqlite3
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger

from .config import get_config


class TrendDirection(Enum):
    """Trend direction classification based on slope analysis."""

    INCREASING = "increasing"  # Positive slope above threshold
    DECREASING = "decreasing"  # Negative slope below threshold
    STABLE = "stable"          # Slope near zero
    VOLATILE = "volatile"      # High variance relative to mean


@dataclass
class TrendDataPoint:
    """Represents a single metric data point in time-series.

    Attributes:
        timestamp: When the metric was recorded
        metric_name: Identifier for the metric type
        value: Numeric value of the metric
        metadata: Additional context (JSON-serializable dict)
    """

    timestamp: datetime
    metric_name: str
    value: float
    metadata: dict[str, Any] | None = None


@dataclass
class TrendAnalysis:
    """Result of trend analysis for a metric over a time window.

    Attributes:
        metric_name: Identifier for the analyzed metric
        trend_direction: Classification of trend (increasing/decreasing/stable/volatile)
        slope: Linear regression slope (change per hour)
        intercept: Linear regression intercept
        forecast: Future predictions {hours_ahead: predicted_value}
        confidence: Confidence score 0-1 (based on R² or data consistency)
        data_points_count: Number of data points analyzed
        window_start: Start of analysis time window
        window_end: End of analysis time window
        mean: Average value over window
        std_dev: Standard deviation over window
        volatility_ratio: std_dev / mean (coefficient of variation)
    """

    metric_name: str
    trend_direction: TrendDirection
    slope: float
    intercept: float
    forecast: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    data_points_count: int = 0
    window_start: datetime | None = None
    window_end: datetime | None = None
    mean: float = 0.0
    std_dev: float = 0.0
    volatility_ratio: float = 0.0


@dataclass
class Anomaly:
    """Represents a detected anomaly in metric data.

    Attributes:
        timestamp: When the anomaly occurred
        metric_name: Identifier for the affected metric
        value: Actual metric value
        expected_value: Expected value based on statistics
        z_score: Standard deviations from mean
        severity: Classification of anomaly severity
        metadata: Additional context from the data point
    """

    timestamp: datetime
    metric_name: str
    value: float
    expected_value: float
    z_score: float
    severity: str  # "low", "medium", "high", "critical"
    metadata: dict[str, Any] | None = None


@dataclass
class PeriodComparison:
    """Result of comparing two time periods for a metric.

    Attributes:
        metric_name: Identifier for the compared metric
        period1_stats: Statistics for first period
        period2_stats: Statistics for second period
        change_pct: Percentage change from period1 to period2
        change_absolute: Absolute change in value
        significant: Whether change is statistically significant
        p_value: Statistical significance (if t-test performed)
    """

    metric_name: str
    period1_stats: dict[str, float]
    period2_stats: dict[str, float]
    change_pct: float
    change_absolute: float
    significant: bool = False
    p_value: float | None = None


class HistoricalTrendAnalyzer:
    """Analyzes historical queue metrics for trends, forecasts, and anomalies.

    This class provides comprehensive statistical analysis of queue performance
    metrics over time. It stores data points in SQLite, performs linear regression
    for trend analysis, forecasts future values, and detects anomalies using
    z-score methods.

    Configuration is loaded from default_configuration.yaml under the
    'trend_analysis' section.
    """

    def __init__(self, db_path: str | None = None):
        """Initialize the historical trend analyzer.

        Args:
            db_path: Path to SQLite database file. If None, uses default from config.
        """
        # Load configuration
        self.enabled = get_config("trend_analysis.enabled", True)
        self.retention_days = get_config("trend_analysis.retention_days", 30)
        self.default_window_hours = get_config("trend_analysis.default_window_hours", 24)
        self.anomaly_sensitivity = get_config("trend_analysis.anomaly_sensitivity", 3.0)
        self.metrics_to_track = get_config("trend_analysis.metrics_to_track", [
            "queue_size", "processing_rate", "error_rate",
            "latency", "success_rate", "resource_usage_cpu", "resource_usage_memory"
        ])

        # Trend thresholds
        self.stable_slope_threshold = get_config("trend_analysis.stable_slope_threshold", 0.1)
        self.volatile_coefficient_threshold = get_config("trend_analysis.volatile_coefficient_threshold", 0.5)

        # Database configuration
        if db_path:
            self.db_path = Path(db_path)
        else:
            # Default to XDG-compliant location
            config_dir = Path.home() / ".config" / "workspace-qdrant"
            config_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = config_dir / "state.db"

        self._conn: sqlite3.Connection | None = None
        self._cleanup_task: asyncio.Task | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize database connection and create tables.

        Loads the metric_history schema and starts the cleanup task.
        """
        if self._initialized:
            return

        # Open database connection
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        # Load and execute schema
        schema_path = Path(__file__).parent / "metric_history_schema.sql"
        if schema_path.exists():
            with open(schema_path) as f:
                schema_sql = f.read()
            self._conn.executescript(schema_sql)
            self._conn.commit()
            logger.debug(f"Initialized metric_history table in {self.db_path}")
        else:
            logger.warning(f"Schema file not found: {schema_path}")
            # Create minimal table if schema file missing
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS metric_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    metadata TEXT
                )
            """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metric_history_name_timestamp
                ON metric_history(metric_name, timestamp DESC)
            """)
            self._conn.commit()

        # Start cleanup task if enabled
        if self.enabled and self.retention_days > 0:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        self._initialized = True
        logger.info(f"HistoricalTrendAnalyzer initialized (retention: {self.retention_days} days)")

    async def close(self) -> None:
        """Close database connection and stop cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        if self._conn:
            self._conn.close()
            self._conn = None

        self._initialized = False

    async def store_metric_point(
        self,
        metric_name: str,
        value: float,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Store a metric data point for historical analysis.

        Args:
            metric_name: Identifier for the metric (e.g., 'queue_size')
            value: Numeric value of the metric
            metadata: Optional additional context (JSON-serializable)

        Raises:
            ValueError: If metric_name is not in configured metrics_to_track
        """
        if not self._initialized:
            await self.initialize()

        if not self.enabled:
            return

        if metric_name not in self.metrics_to_track:
            logger.warning(f"Metric '{metric_name}' not in configured metrics_to_track")
            return

        # Serialize metadata to JSON
        metadata_json = json.dumps(metadata) if metadata else None

        # Insert into database
        cursor = self._conn.cursor()
        cursor.execute("""
            INSERT INTO metric_history (timestamp, metric_name, value, metadata)
            VALUES (?, ?, ?, ?)
        """, (datetime.now(timezone.utc).isoformat(), metric_name, value, metadata_json))
        self._conn.commit()

        logger.debug(f"Stored metric point: {metric_name}={value}")

    async def get_historical_data(
        self,
        metric_name: str,
        hours: int = 168  # 1 week default
    ) -> list[TrendDataPoint]:
        """Retrieve historical data points for a metric.

        Args:
            metric_name: Identifier for the metric
            hours: Number of hours of history to retrieve (default: 168 = 1 week)

        Returns:
            List of TrendDataPoint objects, ordered by timestamp (oldest first)
        """
        if not self._initialized:
            await self.initialize()

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT timestamp, metric_name, value, metadata
            FROM metric_history
            WHERE metric_name = ? AND timestamp >= ?
            ORDER BY timestamp ASC
        """, (metric_name, cutoff_time.isoformat()))

        data_points = []
        for row in cursor.fetchall():
            metadata = json.loads(row['metadata']) if row['metadata'] else None
            data_points.append(TrendDataPoint(
                timestamp=datetime.fromisoformat(row['timestamp']),
                metric_name=row['metric_name'],
                value=float(row['value']),
                metadata=metadata
            ))

        logger.debug(f"Retrieved {len(data_points)} data points for {metric_name} ({hours}h window)")
        return data_points

    def _calculate_linear_regression(
        self,
        data_points: list[TrendDataPoint]
    ) -> tuple[float, float, float]:
        """Calculate linear regression for data points.

        Uses simple linear regression: y = slope * x + intercept
        where x is hours since first data point.

        Args:
            data_points: List of data points (must be ordered by timestamp)

        Returns:
            Tuple of (slope, intercept, r_squared)

        Raises:
            ValueError: If fewer than 2 data points
        """
        if len(data_points) < 2:
            raise ValueError("Need at least 2 data points for linear regression")

        # Convert timestamps to hours since first point
        base_time = data_points[0].timestamp
        x_values = [(dp.timestamp - base_time).total_seconds() / 3600.0 for dp in data_points]
        y_values = [dp.value for dp in data_points]

        n = len(x_values)

        # Calculate means
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n

        # Calculate slope and intercept
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values, strict=False))
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        if denominator == 0:
            # All x values are the same (single timestamp)
            slope = 0.0
            intercept = y_mean
            r_squared = 0.0
        else:
            slope = numerator / denominator
            intercept = y_mean - slope * x_mean

            # Calculate R²
            ss_tot = sum((y - y_mean) ** 2 for y in y_values)
            ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(x_values, y_values, strict=False))
            r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return slope, intercept, r_squared

    def _determine_trend_direction(
        self,
        slope: float,
        values: list[float]
    ) -> TrendDirection:
        """Determine trend direction from slope and volatility.

        Args:
            slope: Linear regression slope
            values: List of metric values

        Returns:
            TrendDirection enum value
        """
        if len(values) < 2:
            return TrendDirection.STABLE

        # Calculate coefficient of variation (volatility)
        mean = statistics.mean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0.0

        # Avoid division by zero
        if mean == 0:
            coefficient_of_variation = 0.0
        else:
            coefficient_of_variation = abs(std_dev / mean)

        # Check for high volatility first
        if coefficient_of_variation > self.volatile_coefficient_threshold:
            return TrendDirection.VOLATILE

        # Check slope magnitude
        abs_slope = abs(slope)
        if abs_slope < self.stable_slope_threshold:
            return TrendDirection.STABLE
        elif slope > 0:
            return TrendDirection.INCREASING
        else:
            return TrendDirection.DECREASING

    async def get_trend_analysis(
        self,
        metric_name: str,
        window_hours: int | None = None
    ) -> TrendAnalysis:
        """Perform trend analysis on a metric over a time window.

        Args:
            metric_name: Identifier for the metric
            window_hours: Time window in hours (default: from config)

        Returns:
            TrendAnalysis object with complete analysis results

        Raises:
            ValueError: If insufficient data for analysis
        """
        if window_hours is None:
            window_hours = self.default_window_hours

        # Get historical data
        data_points = await self.get_historical_data(metric_name, hours=window_hours)

        if len(data_points) < 2:
            raise ValueError(f"Insufficient data for trend analysis: {len(data_points)} points")

        # Extract values
        values = [dp.value for dp in data_points]

        # Calculate statistics
        mean = statistics.mean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
        volatility_ratio = (std_dev / mean) if mean != 0 else 0.0

        # Calculate linear regression
        slope, intercept, r_squared = self._calculate_linear_regression(data_points)

        # Determine trend direction
        trend_direction = self._determine_trend_direction(slope, values)

        # Generate forecasts for 1, 6, 12, 24 hours ahead
        current_hours = (data_points[-1].timestamp - data_points[0].timestamp).total_seconds() / 3600.0
        forecast = {}
        for hours_ahead in [1, 6, 12, 24]:
            future_hours = current_hours + hours_ahead
            forecast_value = slope * future_hours + intercept
            forecast[f"{hours_ahead}h"] = max(0.0, forecast_value)  # No negative forecasts

        return TrendAnalysis(
            metric_name=metric_name,
            trend_direction=trend_direction,
            slope=slope,
            intercept=intercept,
            forecast=forecast,
            confidence=r_squared,
            data_points_count=len(data_points),
            window_start=data_points[0].timestamp,
            window_end=data_points[-1].timestamp,
            mean=mean,
            std_dev=std_dev,
            volatility_ratio=volatility_ratio
        )

    async def forecast_metric(
        self,
        metric_name: str,
        hours_ahead: float = 1.0
    ) -> float:
        """Forecast future metric value using linear extrapolation.

        Args:
            metric_name: Identifier for the metric
            hours_ahead: Number of hours into the future to forecast

        Returns:
            Forecasted metric value (minimum 0.0)
        """
        analysis = await self.get_trend_analysis(metric_name)

        # Calculate hours from start of window
        current_hours = (analysis.window_end - analysis.window_start).total_seconds() / 3600.0
        future_hours = current_hours + hours_ahead

        # Linear extrapolation
        forecast_value = analysis.slope * future_hours + analysis.intercept

        return max(0.0, forecast_value)

    def _calculate_z_scores(
        self,
        values: list[float]
    ) -> list[float]:
        """Calculate z-scores for a list of values.

        Args:
            values: List of numeric values

        Returns:
            List of z-scores (number of standard deviations from mean)
        """
        if len(values) < 2:
            return [0.0] * len(values)

        mean = statistics.mean(values)
        std_dev = statistics.stdev(values)

        if std_dev == 0:
            return [0.0] * len(values)

        return [(v - mean) / std_dev for v in values]

    async def detect_anomalies(
        self,
        metric_name: str,
        sensitivity: float | None = None,
        window_hours: int | None = None
    ) -> list[Anomaly]:
        """Detect anomalies in metric data using z-score method.

        Args:
            metric_name: Identifier for the metric
            sensitivity: Z-score threshold (default: from config, typically 3.0)
            window_hours: Time window in hours (default: from config)

        Returns:
            List of detected Anomaly objects
        """
        if sensitivity is None:
            sensitivity = self.anomaly_sensitivity

        if window_hours is None:
            window_hours = self.default_window_hours

        # Get historical data
        data_points = await self.get_historical_data(metric_name, hours=window_hours)

        if len(data_points) < 2:
            return []

        # Calculate statistics
        values = [dp.value for dp in data_points]
        mean = statistics.mean(values)
        z_scores = self._calculate_z_scores(values)

        # Detect anomalies
        anomalies = []
        for dp, z_score in zip(data_points, z_scores, strict=False):
            if abs(z_score) > sensitivity:
                # Classify severity
                abs_z = abs(z_score)
                if abs_z > sensitivity * 2:
                    severity = "critical"
                elif abs_z > sensitivity * 1.5:
                    severity = "high"
                elif abs_z > sensitivity * 1.2:
                    severity = "medium"
                else:
                    severity = "low"

                anomalies.append(Anomaly(
                    timestamp=dp.timestamp,
                    metric_name=metric_name,
                    value=dp.value,
                    expected_value=mean,
                    z_score=z_score,
                    severity=severity,
                    metadata=dp.metadata
                ))

        logger.info(f"Detected {len(anomalies)} anomalies in {metric_name} (sensitivity={sensitivity})")
        return anomalies

    def _calculate_period_stats(
        self,
        data_points: list[TrendDataPoint]
    ) -> dict[str, float]:
        """Calculate statistical summary for a period.

        Args:
            data_points: List of data points for the period

        Returns:
            Dict with keys: mean, median, std_dev, min, max, count
        """
        if not data_points:
            return {
                "mean": 0.0,
                "median": 0.0,
                "std_dev": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0
            }

        values = [dp.value for dp in data_points]

        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "count": len(values)
        }

    async def compare_periods(
        self,
        metric_name: str,
        period1_hours: int,
        period2_hours: int,
        period2_offset_hours: int = 0
    ) -> PeriodComparison:
        """Compare two time periods for a metric.

        Args:
            metric_name: Identifier for the metric
            period1_hours: Duration of first period (hours)
            period2_hours: Duration of second period (hours)
            period2_offset_hours: Hours before period1 where period2 starts (default: 0 = immediately before period1)

        Returns:
            PeriodComparison object with statistical comparison

        Example:
            # Compare last 24h vs previous 24h
            comparison = await analyzer.compare_periods("queue_size", 24, 24, 0)
        """
        # Get period 1 data (most recent)
        period1_data = await self.get_historical_data(metric_name, hours=period1_hours)

        # Get period 2 data (earlier period)
        # Calculate time range for period 2
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=period1_hours + period2_offset_hours)
        start_time = cutoff_time - timedelta(hours=period2_hours)

        cursor = self._conn.cursor()
        cursor.execute("""
            SELECT timestamp, metric_name, value, metadata
            FROM metric_history
            WHERE metric_name = ? AND timestamp >= ? AND timestamp < ?
            ORDER BY timestamp ASC
        """, (metric_name, start_time.isoformat(), cutoff_time.isoformat()))

        period2_data = []
        for row in cursor.fetchall():
            metadata = json.loads(row['metadata']) if row['metadata'] else None
            period2_data.append(TrendDataPoint(
                timestamp=datetime.fromisoformat(row['timestamp']),
                metric_name=row['metric_name'],
                value=float(row['value']),
                metadata=metadata
            ))

        # Calculate statistics for both periods
        period1_stats = self._calculate_period_stats(period1_data)
        period2_stats = self._calculate_period_stats(period2_data)

        # Calculate change
        if period2_stats["mean"] != 0:
            change_pct = ((period1_stats["mean"] - period2_stats["mean"]) / period2_stats["mean"]) * 100
        else:
            change_pct = 0.0 if period1_stats["mean"] == 0 else float('inf')

        change_absolute = period1_stats["mean"] - period2_stats["mean"]

        # Simple significance test: change > 2 * combined std error
        if period1_stats["count"] > 1 and period2_stats["count"] > 1:
            se1 = period1_stats["std_dev"] / (period1_stats["count"] ** 0.5)
            se2 = period2_stats["std_dev"] / (period2_stats["count"] ** 0.5)
            combined_se = (se1 ** 2 + se2 ** 2) ** 0.5
            significant = abs(change_absolute) > (2 * combined_se)
        else:
            significant = False

        return PeriodComparison(
            metric_name=metric_name,
            period1_stats=period1_stats,
            period2_stats=period2_stats,
            change_pct=change_pct,
            change_absolute=change_absolute,
            significant=significant,
            p_value=None  # Could implement proper t-test if scipy available
        )

    async def export_trends(
        self,
        format: str = 'json'
    ) -> str:
        """Export trend analyses for all tracked metrics.

        Args:
            format: Export format ('json' only for now)

        Returns:
            JSON string with trend analyses for all metrics
        """
        if format != 'json':
            raise ValueError(f"Unsupported export format: {format}")

        export_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "retention_days": self.retention_days,
            "window_hours": self.default_window_hours,
            "metrics": {}
        }

        for metric_name in self.metrics_to_track:
            try:
                analysis = await self.get_trend_analysis(metric_name)
                export_data["metrics"][metric_name] = {
                    "trend_direction": analysis.trend_direction.value,
                    "slope": analysis.slope,
                    "intercept": analysis.intercept,
                    "forecast": analysis.forecast,
                    "confidence": analysis.confidence,
                    "data_points_count": analysis.data_points_count,
                    "window_start": analysis.window_start.isoformat() if analysis.window_start else None,
                    "window_end": analysis.window_end.isoformat() if analysis.window_end else None,
                    "mean": analysis.mean,
                    "std_dev": analysis.std_dev,
                    "volatility_ratio": analysis.volatility_ratio
                }
            except ValueError as e:
                # Insufficient data for this metric
                export_data["metrics"][metric_name] = {
                    "error": str(e)
                }

        return json.dumps(export_data, indent=2)

    async def _cleanup_old_data(self) -> int:
        """Remove metric data older than retention period.

        Returns:
            Number of rows deleted
        """
        if not self._initialized or self.retention_days <= 0:
            return 0

        cutoff_time = datetime.now(timezone.utc) - timedelta(days=self.retention_days)

        cursor = self._conn.cursor()
        cursor.execute("""
            DELETE FROM metric_history
            WHERE timestamp < ?
        """, (cutoff_time.isoformat(),))

        deleted_count = cursor.rowcount
        self._conn.commit()

        # Vacuum to reclaim space
        self._conn.execute("VACUUM")

        logger.info(f"Cleaned up {deleted_count} old metric records (retention: {self.retention_days} days)")
        return deleted_count

    async def _cleanup_loop(self) -> None:
        """Background task that periodically cleans up old data."""
        while True:
            try:
                # Run cleanup once per day
                await asyncio.sleep(86400)  # 24 hours
                await self._cleanup_old_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour on error
