"""
Unit tests for queue backpressure detection and alerting.

Tests cover:
- Backpressure indicator calculation
- Drain time estimation
- Queue size prediction
- Alert generation at different severity levels
- Threshold configuration
- Callback registration and triggering
- Background monitoring task
- Edge cases (empty queues, stable queues, rapid growth)
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.python.common.core.queue_backpressure import (
    BackpressureAlert,
    BackpressureDetector,
    BackpressureIndicators,
    BackpressureSeverity,
    BackpressureThresholds,
    QueueTrend,
)
from src.python.common.core.queue_statistics import (
    QueueStatistics,
    QueueStatisticsCollector,
)


@pytest.fixture
def mock_stats_collector():
    """Create mock statistics collector."""
    collector = AsyncMock(spec=QueueStatisticsCollector)
    collector._initialized = True
    return collector


@pytest.fixture
def default_thresholds():
    """Create default thresholds for testing."""
    return BackpressureThresholds()


@pytest.fixture
async def detector(mock_stats_collector, default_thresholds):
    """Create backpressure detector with mock collector."""
    detector = BackpressureDetector(
        stats_collector=mock_stats_collector, thresholds=default_thresholds
    )
    await detector.initialize()
    yield detector
    await detector.close()


class TestBackpressureIndicators:
    """Test BackpressureIndicators dataclass."""

    def test_indicators_creation(self):
        """Test creating backpressure indicators."""
        indicators = BackpressureIndicators(
            queue_growth_rate=50.0,
            processing_capacity_used=75.0,
            estimated_drain_time=timedelta(minutes=45),
            queue_size=1000,
            queue_trend=QueueTrend.GROWING,
        )

        assert indicators.queue_growth_rate == 50.0
        assert indicators.processing_capacity_used == 75.0
        assert indicators.estimated_drain_time == timedelta(minutes=45)
        assert indicators.queue_size == 1000
        assert indicators.queue_trend == QueueTrend.GROWING

    def test_indicators_to_dict(self):
        """Test converting indicators to dictionary."""
        indicators = BackpressureIndicators(
            queue_growth_rate=50.0,
            processing_capacity_used=75.0,
            estimated_drain_time=timedelta(minutes=45),
            queue_size=1000,
            queue_trend=QueueTrend.GROWING,
        )

        result = indicators.to_dict()

        assert result["queue_growth_rate"] == 50.0
        assert result["processing_capacity_used"] == 75.0
        assert result["estimated_drain_time_seconds"] == 2700  # 45 minutes
        assert result["estimated_drain_time_human"] == "45m"
        assert result["queue_size"] == 1000
        assert result["queue_trend"] == "growing"
        assert "timestamp" in result

    def test_format_timedelta(self):
        """Test timedelta formatting for human readability."""
        # Test seconds
        td = timedelta(seconds=30)
        formatted = BackpressureIndicators._format_timedelta(td)
        assert formatted == "30s"

        # Test minutes
        td = timedelta(minutes=25)
        formatted = BackpressureIndicators._format_timedelta(td)
        assert formatted == "25m"

        # Test hours
        td = timedelta(hours=2, minutes=15)
        formatted = BackpressureIndicators._format_timedelta(td)
        assert formatted == "2h 15m"

        # Test days
        td = timedelta(days=1, hours=6)
        formatted = BackpressureIndicators._format_timedelta(td)
        assert formatted == "1d 6h"


class TestBackpressureAlert:
    """Test BackpressureAlert dataclass."""

    def test_alert_creation(self):
        """Test creating backpressure alert."""
        indicators = BackpressureIndicators(
            queue_growth_rate=100.0,
            processing_capacity_used=85.0,
            estimated_drain_time=timedelta(hours=1),
            queue_size=2000,
            queue_trend=QueueTrend.GROWING,
        )

        alert = BackpressureAlert(
            severity=BackpressureSeverity.HIGH,
            indicators=indicators,
            recommended_actions=["Scale up workers", "Investigate bottlenecks"],
        )

        assert alert.severity == BackpressureSeverity.HIGH
        assert alert.indicators == indicators
        assert len(alert.recommended_actions) == 2

    def test_alert_to_dict(self):
        """Test converting alert to dictionary."""
        indicators = BackpressureIndicators(
            queue_growth_rate=100.0,
            processing_capacity_used=85.0,
            estimated_drain_time=timedelta(hours=1),
            queue_size=2000,
            queue_trend=QueueTrend.GROWING,
        )

        alert = BackpressureAlert(
            severity=BackpressureSeverity.HIGH,
            indicators=indicators,
            recommended_actions=["Scale up workers"],
        )

        result = alert.to_dict()

        assert result["severity"] == "high"
        assert "indicators" in result
        assert result["recommended_actions"] == ["Scale up workers"]
        assert "timestamp" in result


class TestBackpressureDetector:
    """Test BackpressureDetector class."""

    @pytest.mark.asyncio
    async def test_initialization(self, mock_stats_collector, default_thresholds):
        """Test detector initialization."""
        detector = BackpressureDetector(
            stats_collector=mock_stats_collector, thresholds=default_thresholds
        )

        assert not detector._initialized

        await detector.initialize()

        assert detector._initialized

        await detector.close()

    @pytest.mark.asyncio
    async def test_calculate_drain_time_normal(self, detector, mock_stats_collector):
        """Test drain time calculation with normal processing rate."""
        # Mock statistics with 100 items/min processing rate, 500 items in queue
        stats = QueueStatistics(queue_size=500, processing_rate=100.0)
        mock_stats_collector.get_current_statistics.return_value = stats

        drain_time = await detector.calculate_drain_time()

        # Expected: 500 items / 100 items/min = 5 minutes
        assert drain_time == timedelta(minutes=5)

    @pytest.mark.asyncio
    async def test_calculate_drain_time_zero_rate(self, detector, mock_stats_collector):
        """Test drain time calculation with zero processing rate."""
        stats = QueueStatistics(queue_size=500, processing_rate=0.0)
        mock_stats_collector.get_current_statistics.return_value = stats

        drain_time = await detector.calculate_drain_time()

        # Expected: infinite time (capped at 365 days)
        assert drain_time == timedelta(days=365)

    @pytest.mark.asyncio
    async def test_calculate_drain_time_empty_queue(
        self, detector, mock_stats_collector
    ):
        """Test drain time calculation with empty queue."""
        stats = QueueStatistics(queue_size=0, processing_rate=100.0)
        mock_stats_collector.get_current_statistics.return_value = stats

        drain_time = await detector.calculate_drain_time()

        # Expected: 0 minutes
        assert drain_time == timedelta(minutes=0)

    @pytest.mark.asyncio
    async def test_predict_queue_size(self, detector, mock_stats_collector):
        """Test queue size prediction based on growth rate."""
        # Mock current statistics
        stats = QueueStatistics(queue_size=1000, processing_rate=50.0)
        mock_stats_collector.get_current_statistics.return_value = stats

        # Mock queue size history to calculate growth rate
        # Simulate growth of 20 items/min
        async with detector._lock:
            now = datetime.now(timezone.utc)
            detector._queue_size_history.append((now - timedelta(minutes=5), 900))
            detector._queue_size_history.append((now, 1000))

        # Predict 30 minutes ahead
        predicted_size = await detector.predict_queue_size(minutes_ahead=30)

        # Growth rate: (1000 - 900) / 5 = 20 items/min
        # Predicted: 1000 + (20 * 30) = 1600
        assert predicted_size == 1600

    @pytest.mark.asyncio
    async def test_predict_queue_size_negative_growth(
        self, detector, mock_stats_collector
    ):
        """Test queue size prediction with negative growth (draining)."""
        stats = QueueStatistics(queue_size=500, processing_rate=50.0)
        mock_stats_collector.get_current_statistics.return_value = stats

        # Mock queue size history showing drain
        async with detector._lock:
            now = datetime.now(timezone.utc)
            detector._queue_size_history.append((now - timedelta(minutes=5), 1000))
            detector._queue_size_history.append((now, 500))

        # Predict 30 minutes ahead
        predicted_size = await detector.predict_queue_size(minutes_ahead=30)

        # Growth rate: (500 - 1000) / 5 = -100 items/min
        # Predicted: 500 + (-100 * 30) = -2500, capped at 0
        assert predicted_size == 0

    @pytest.mark.asyncio
    async def test_determine_trend_growing(self, detector):
        """Test queue trend determination for growing queue."""
        trend = detector._determine_trend(growth_rate=20.0)
        assert trend == QueueTrend.GROWING

    @pytest.mark.asyncio
    async def test_determine_trend_draining(self, detector):
        """Test queue trend determination for draining queue."""
        trend = detector._determine_trend(growth_rate=-20.0)
        assert trend == QueueTrend.DRAINING

    @pytest.mark.asyncio
    async def test_determine_trend_stable(self, detector):
        """Test queue trend determination for stable queue."""
        trend = detector._determine_trend(growth_rate=2.0)
        assert trend == QueueTrend.STABLE

    @pytest.mark.asyncio
    async def test_calculate_severity_none(self, detector):
        """Test severity calculation for no backpressure."""
        indicators = BackpressureIndicators(
            queue_growth_rate=5.0,  # Below LOW threshold
            processing_capacity_used=30.0,
            estimated_drain_time=timedelta(minutes=5),
            queue_size=100,
            queue_trend=QueueTrend.GROWING,
        )

        severity = detector._calculate_severity(indicators)
        assert severity == BackpressureSeverity.NONE

    @pytest.mark.asyncio
    async def test_calculate_severity_low(self, detector):
        """Test severity calculation for LOW backpressure."""
        indicators = BackpressureIndicators(
            queue_growth_rate=15.0,  # Above LOW threshold
            processing_capacity_used=40.0,
            estimated_drain_time=timedelta(minutes=20),
            queue_size=300,
            queue_trend=QueueTrend.GROWING,
        )

        severity = detector._calculate_severity(indicators)
        assert severity == BackpressureSeverity.LOW

    @pytest.mark.asyncio
    async def test_calculate_severity_medium(self, detector):
        """Test severity calculation for MEDIUM backpressure."""
        indicators = BackpressureIndicators(
            queue_growth_rate=60.0,  # Above MEDIUM threshold
            processing_capacity_used=60.0,
            estimated_drain_time=timedelta(minutes=45),  # 30-60 min range
            queue_size=1000,
            queue_trend=QueueTrend.GROWING,
        )

        severity = detector._calculate_severity(indicators)
        assert severity == BackpressureSeverity.MEDIUM

    @pytest.mark.asyncio
    async def test_calculate_severity_high(self, detector):
        """Test severity calculation for HIGH backpressure."""
        indicators = BackpressureIndicators(
            queue_growth_rate=120.0,  # Above HIGH threshold
            processing_capacity_used=75.0,
            estimated_drain_time=timedelta(minutes=90),  # 60-120 min range
            queue_size=2000,
            queue_trend=QueueTrend.GROWING,
        )

        severity = detector._calculate_severity(indicators)
        assert severity == BackpressureSeverity.HIGH

    @pytest.mark.asyncio
    async def test_calculate_severity_critical_growth(self, detector):
        """Test severity calculation for CRITICAL backpressure (high growth)."""
        indicators = BackpressureIndicators(
            queue_growth_rate=250.0,  # Above CRITICAL threshold
            processing_capacity_used=90.0,
            estimated_drain_time=timedelta(minutes=60),
            queue_size=5000,
            queue_trend=QueueTrend.GROWING,
        )

        severity = detector._calculate_severity(indicators)
        assert severity == BackpressureSeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_calculate_severity_critical_drain_time(self, detector):
        """Test severity calculation for CRITICAL backpressure (long drain time)."""
        indicators = BackpressureIndicators(
            queue_growth_rate=50.0,
            processing_capacity_used=80.0,
            estimated_drain_time=timedelta(hours=3),  # Above CRITICAL threshold
            queue_size=3000,
            queue_trend=QueueTrend.GROWING,
        )

        severity = detector._calculate_severity(indicators)
        assert severity == BackpressureSeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_get_recommended_actions_low(self, detector):
        """Test recommended actions for LOW severity."""
        indicators = BackpressureIndicators(
            queue_growth_rate=15.0,
            processing_capacity_used=40.0,
            estimated_drain_time=timedelta(minutes=20),
            queue_size=300,
            queue_trend=QueueTrend.GROWING,
        )

        actions = detector._get_recommended_actions(BackpressureSeverity.LOW, indicators)

        assert len(actions) > 0
        assert any("monitor" in action.lower() for action in actions)

    @pytest.mark.asyncio
    async def test_get_recommended_actions_critical(self, detector):
        """Test recommended actions for CRITICAL severity."""
        indicators = BackpressureIndicators(
            queue_growth_rate=250.0,
            processing_capacity_used=90.0,
            estimated_drain_time=timedelta(hours=3),
            queue_size=5000,
            queue_trend=QueueTrend.GROWING,
        )

        actions = detector._get_recommended_actions(
            BackpressureSeverity.CRITICAL, indicators
        )

        assert len(actions) > 0
        assert any("emergency" in action.lower() for action in actions)

    @pytest.mark.asyncio
    async def test_detect_backpressure_none(self, detector, mock_stats_collector):
        """Test backpressure detection with no backpressure."""
        stats = QueueStatistics(queue_size=100, processing_rate=100.0)
        mock_stats_collector.get_current_statistics.return_value = stats

        # Mock minimal growth
        async with detector._lock:
            now = datetime.now(timezone.utc)
            detector._queue_size_history.append((now - timedelta(minutes=5), 95))
            detector._queue_size_history.append((now, 100))

        alert = await detector.detect_backpressure()

        assert alert is None

    @pytest.mark.asyncio
    async def test_detect_backpressure_alert(self, detector, mock_stats_collector):
        """Test backpressure detection with alert generation."""
        stats = QueueStatistics(queue_size=1000, processing_rate=50.0)
        mock_stats_collector.get_current_statistics.return_value = stats

        # Mock significant growth rate
        async with detector._lock:
            now = datetime.now(timezone.utc)
            detector._queue_size_history.append((now - timedelta(minutes=5), 700))
            detector._queue_size_history.append((now, 1000))

        alert = await detector.detect_backpressure()

        assert alert is not None
        assert alert.severity in [
            BackpressureSeverity.LOW,
            BackpressureSeverity.MEDIUM,
            BackpressureSeverity.HIGH,
        ]
        assert len(alert.recommended_actions) > 0

    @pytest.mark.asyncio
    async def test_callback_registration(self, detector):
        """Test callback registration."""
        callback = AsyncMock()

        result = await detector.register_backpressure_callback(callback)

        assert result is True
        assert callback in detector._callbacks

    @pytest.mark.asyncio
    async def test_callback_duplicate_registration(self, detector):
        """Test duplicate callback registration."""
        callback = AsyncMock()

        await detector.register_backpressure_callback(callback)
        result = await detector.register_backpressure_callback(callback)

        assert result is False

    @pytest.mark.asyncio
    async def test_callback_unregistration(self, detector):
        """Test callback unregistration."""
        callback = AsyncMock()

        await detector.register_backpressure_callback(callback)
        result = await detector.unregister_backpressure_callback(callback)

        assert result is True
        assert callback not in detector._callbacks

    @pytest.mark.asyncio
    async def test_callback_triggering(self, detector, mock_stats_collector):
        """Test that callbacks are triggered on alert."""
        callback = AsyncMock()
        await detector.register_backpressure_callback(callback)

        # Create conditions for backpressure alert
        stats = QueueStatistics(queue_size=1000, processing_rate=50.0)
        mock_stats_collector.get_current_statistics.return_value = stats

        async with detector._lock:
            now = datetime.now(timezone.utc)
            detector._queue_size_history.append((now - timedelta(minutes=5), 700))
            detector._queue_size_history.append((now, 1000))

        # Trigger detection
        alert = await detector.detect_backpressure()

        # Manually trigger callbacks (monitoring loop does this)
        if alert:
            await detector._trigger_callbacks(alert)

        # Verify callback was called
        callback.assert_called_once()
        assert isinstance(callback.call_args[0][0], BackpressureAlert)

    @pytest.mark.asyncio
    async def test_monitoring_start_stop(self, detector):
        """Test starting and stopping monitoring."""
        # Start monitoring
        result = await detector.start_monitoring(check_interval_seconds=1)
        assert result is True
        assert detector._monitoring_task is not None
        assert not detector._monitoring_task.done()

        # Try to start again (should fail)
        result = await detector.start_monitoring(check_interval_seconds=1)
        assert result is False

        # Stop monitoring
        result = await detector.stop_monitoring()
        assert result is True

        # Try to stop again (should fail)
        result = await detector.stop_monitoring()
        assert result is False

    @pytest.mark.asyncio
    async def test_monitoring_loop_integration(self, detector, mock_stats_collector):
        """Test monitoring loop integration with callback."""
        callback_invoked = asyncio.Event()
        received_alerts = []

        async def test_callback(alert):
            received_alerts.append(alert)
            callback_invoked.set()

        await detector.register_backpressure_callback(test_callback)

        # Create backpressure conditions
        stats = QueueStatistics(queue_size=1000, processing_rate=50.0)
        mock_stats_collector.get_current_statistics.return_value = stats

        async with detector._lock:
            now = datetime.now(timezone.utc)
            detector._queue_size_history.append((now - timedelta(minutes=5), 700))
            detector._queue_size_history.append((now, 1000))

        # Start monitoring with short interval
        await detector.start_monitoring(check_interval_seconds=1)

        # Wait for callback to be invoked
        try:
            await asyncio.wait_for(callback_invoked.wait(), timeout=3.0)
        except asyncio.TimeoutError:
            pytest.fail("Callback was not invoked within timeout")
        finally:
            await detector.stop_monitoring()

        # Verify alert was received
        assert len(received_alerts) > 0
        assert isinstance(received_alerts[0], BackpressureAlert)

    @pytest.mark.asyncio
    async def test_capacity_calculation(self, detector):
        """Test processing capacity utilization calculation."""
        # Test within capacity
        capacity = await detector._calculate_capacity_used(processing_rate=500.0)
        assert capacity == 50.0  # 500/1000 * 100

        # Test at capacity
        capacity = await detector._calculate_capacity_used(processing_rate=1000.0)
        assert capacity == 100.0

        # Test over capacity (capped at 100%)
        capacity = await detector._calculate_capacity_used(processing_rate=1500.0)
        assert capacity == 100.0

        # Test zero rate
        capacity = await detector._calculate_capacity_used(processing_rate=0.0)
        assert capacity == 0.0

    @pytest.mark.asyncio
    async def test_growth_rate_calculation(self, detector):
        """Test queue growth rate calculation from history."""
        # Add queue size history
        async with detector._lock:
            now = datetime.now(timezone.utc)
            detector._queue_size_history.append((now - timedelta(minutes=10), 500))
            detector._queue_size_history.append((now - timedelta(minutes=5), 700))
            detector._queue_size_history.append((now, 1000))

        growth_rate = await detector._calculate_growth_rate()

        # Expected: (1000 - 500) / 10 minutes = 50 items/min
        assert growth_rate == pytest.approx(50.0, rel=0.1)

    @pytest.mark.asyncio
    async def test_growth_rate_insufficient_history(self, detector):
        """Test growth rate calculation with insufficient history."""
        # Empty history
        growth_rate = await detector._calculate_growth_rate()
        assert growth_rate == 0.0

        # Single data point
        async with detector._lock:
            detector._queue_size_history.append((datetime.now(timezone.utc), 100))

        growth_rate = await detector._calculate_growth_rate()
        assert growth_rate == 0.0

    @pytest.mark.asyncio
    async def test_custom_thresholds(self, mock_stats_collector):
        """Test detector with custom thresholds."""
        custom_thresholds = BackpressureThresholds(
            low_growth_rate=5.0,
            medium_growth_rate=25.0,
            high_growth_rate=50.0,
            critical_growth_rate=100.0,
            max_processing_capacity=500.0,
        )

        detector = BackpressureDetector(
            stats_collector=mock_stats_collector, thresholds=custom_thresholds
        )
        await detector.initialize()

        try:
            # Test with custom thresholds
            indicators = BackpressureIndicators(
                queue_growth_rate=30.0,
                processing_capacity_used=60.0,
                estimated_drain_time=timedelta(minutes=45),
                queue_size=800,
                queue_trend=QueueTrend.GROWING,
            )

            severity = detector._calculate_severity(indicators)
            # With custom thresholds, 30.0 growth should be MEDIUM
            assert severity in [BackpressureSeverity.MEDIUM, BackpressureSeverity.HIGH]

        finally:
            await detector.close()
