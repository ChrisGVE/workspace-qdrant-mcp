"""
Unit tests for queue health status calculation.

Basic tests covering core functionality.
"""

import pytest
from src.python.common.core.queue_health import (
    HealthIndicator,
    HealthStatus,
    HealthThresholds,
    HealthWeights,
    QueueHealthCalculator,
    QueueHealthStatus,
)


class TestHealthStatus:
    """Test HealthStatus enum."""

    def test_health_status_values(self):
        """Test health status enum values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.CRITICAL.value == "critical"


class TestHealthIndicator:
    """Test HealthIndicator dataclass."""

    def test_indicator_creation(self):
        """Test creating a health indicator."""
        indicator = HealthIndicator(
            name="test_indicator",
            status=HealthStatus.HEALTHY,
            value=95.0,
            threshold=80.0,
            message="Test indicator is healthy",
            score=100.0,
        )

        assert indicator.name == "test_indicator"
        assert indicator.status == HealthStatus.HEALTHY
        assert indicator.value == 95.0
        assert indicator.threshold == 80.0
        assert indicator.message == "Test indicator is healthy"
        assert indicator.score == 100.0

    def test_indicator_to_dict(self):
        """Test converting indicator to dictionary."""
        indicator = HealthIndicator(
            name="test_indicator",
            status=HealthStatus.DEGRADED,
            value=75.5,
            threshold=80.0,
            message="Below threshold",
            score=70.0,
        )

        result = indicator.to_dict()

        assert result["name"] == "test_indicator"
        assert result["status"] == "degraded"
        assert result["value"] == 75.5
        assert result["threshold"] == 80.0
        assert result["message"] == "Below threshold"
        assert result["score"] == 70.0


class TestQueueHealthStatus:
    """Test QueueHealthStatus dataclass."""

    def test_health_status_creation(self):
        """Test creating a queue health status."""
        indicators = [
            HealthIndicator(
                name="test1",
                status=HealthStatus.HEALTHY,
                value=100.0,
                threshold=80.0,
                message="Good",
                score=100.0,
            ),
            HealthIndicator(
                name="test2",
                status=HealthStatus.DEGRADED,
                value=70.0,
                threshold=80.0,
                message="Below threshold",
                score=70.0,
            ),
        ]

        health_status = QueueHealthStatus(
            overall_status=HealthStatus.DEGRADED,
            indicators=indicators,
            score=85.0,
            recommendations=["Monitor closely", "Consider scaling"],
        )

        assert health_status.overall_status == HealthStatus.DEGRADED
        assert len(health_status.indicators) == 2
        assert health_status.score == 85.0
        assert len(health_status.recommendations) == 2

    def test_health_status_to_dict(self):
        """Test converting health status to dictionary."""
        indicators = [
            HealthIndicator(
                name="test",
                status=HealthStatus.HEALTHY,
                value=100.0,
                threshold=80.0,
                message="Good",
                score=100.0,
            )
        ]

        health_status = QueueHealthStatus(
            overall_status=HealthStatus.HEALTHY,
            indicators=indicators,
            score=100.0,
            recommendations=["Keep monitoring"],
        )

        result = health_status.to_dict()

        assert result["overall_status"] == "healthy"
        assert result["score"] == 100.0
        assert len(result["indicators"]) == 1
        assert result["recommendations"] == ["Keep monitoring"]


class TestHealthThresholds:
    """Test HealthThresholds dataclass."""

    def test_default_thresholds(self):
        """Test default threshold values."""
        thresholds = HealthThresholds()

        assert thresholds.backlog_normal == 1000
        assert thresholds.backlog_warning == 5000
        assert thresholds.backlog_critical == 10000
        assert thresholds.processing_rate_min == 10.0
        assert thresholds.error_rate_max == 5.0

    def test_custom_thresholds(self):
        """Test custom threshold values."""
        thresholds = HealthThresholds(
            backlog_normal=2000,
            backlog_warning=8000,
            backlog_critical=15000,
            processing_rate_min=20.0,
            error_rate_max=3.0,
        )

        assert thresholds.backlog_normal == 2000
        assert thresholds.backlog_warning == 8000
        assert thresholds.backlog_critical == 15000
        assert thresholds.processing_rate_min == 20.0
        assert thresholds.error_rate_max == 3.0


class TestHealthWeights:
    """Test HealthWeights dataclass."""

    def test_default_weights(self):
        """Test default weight values."""
        weights = HealthWeights()

        assert weights.backlog == 20.0
        assert weights.processing_rate == 20.0
        assert weights.error_rate == 25.0
        assert weights.latency == 15.0
        assert weights.success_rate == 20.0
        assert weights.backpressure == 0.0
        assert weights.resource_usage == 0.0

    def test_custom_weights(self):
        """Test custom weight values."""
        weights = HealthWeights(
            backlog=30.0,
            processing_rate=30.0,
            error_rate=20.0,
            latency=10.0,
            success_rate=10.0,
        )

        assert weights.backlog == 30.0
        assert weights.processing_rate == 30.0
        assert weights.error_rate == 20.0
        assert weights.latency == 10.0
        assert weights.success_rate == 10.0


class TestHealthCalculatorBasics:
    """Test basic QueueHealthCalculator functionality."""

    def test_determine_health_status(self):
        """Test health status determination from score."""
        # Create a minimal calculator to test internal methods
        from unittest.mock import AsyncMock
        from src.python.common.core.queue_statistics import QueueStatisticsCollector

        mock_collector = AsyncMock(spec=QueueStatisticsCollector)
        calculator = QueueHealthCalculator(stats_collector=mock_collector)

        assert calculator._determine_health_status(85.0) == HealthStatus.HEALTHY
        assert calculator._determine_health_status(70.0) == HealthStatus.DEGRADED
        assert calculator._determine_health_status(50.0) == HealthStatus.UNHEALTHY
        assert calculator._determine_health_status(30.0) == HealthStatus.CRITICAL

    def test_register_custom_check_validation(self):
        """Test custom health check registration validation."""
        from unittest.mock import AsyncMock
        from src.python.common.core.queue_statistics import QueueStatisticsCollector

        mock_collector = AsyncMock(spec=QueueStatisticsCollector)
        calculator = QueueHealthCalculator(stats_collector=mock_collector)

        # Test invalid callable
        with pytest.raises(ValueError):
            calculator.register_health_check(
                name="invalid", check_func="not_callable", weight=10.0
            )

        # Test negative weight
        async def custom_check():
            return (HealthStatus.HEALTHY, 100.0, 80.0, "OK")

        with pytest.raises(ValueError):
            calculator.register_health_check(
                name="negative_weight", check_func=custom_check, weight=-10.0
            )
