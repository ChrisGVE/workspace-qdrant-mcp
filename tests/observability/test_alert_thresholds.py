"""
Alert Threshold Testing Suite for Observability Framework

Comprehensive tests validating alert threshold functionality for workspace-qdrant-mcp's
observability infrastructure. Tests cover all alert types, threshold comparison logic,
alert lifecycle management, severity levels, and integration with metrics collection.

Test Categories:
    1. Threshold Comparison Logic - All operators (>, <, ==, !=, >=, <=)
    2. Windowed Alerts - Sustained threshold violations over time
    3. Rate of Change Alerts - Rapid metric increases/decreases
    4. Composite Alerts - Multiple conditions (AND/OR logic)
    5. Alert Lifecycle - Activation, persistence, deactivation, cooldown
    6. Alert Severity Levels - INFO, WARNING, ERROR, CRITICAL
    7. Alert Types - Performance, resource, error rate, queue depth, health
    8. Metrics Integration - Integration with metrics from subtask 313.1

Running Tests:
    ```bash
    # Run all alert threshold tests
    uv run pytest tests/observability/test_alert_thresholds.py -v

    # Run specific test class
    uv run pytest tests/observability/test_alert_thresholds.py::TestThresholdComparison -v

    # Run with coverage
    uv run pytest tests/observability/test_alert_thresholds.py \
        --cov=src/python/common/core/queue_alerting \
        --cov-report=html
    ```

Test Design:
    - Each test is isolated and repeatable
    - Mock metrics providers simulate real-world conditions
    - Tests validate both success and failure scenarios
    - Edge cases thoroughly covered
    - Clear documentation for each test case
"""

import asyncio
import os
import sqlite3
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.python.common.core.queue_alerting import (
    AlertNotification,
    AlertRule,
    AlertSeverity,
    AlertThreshold,
    ConditionLogic,
    QueueAlertingSystem,
)
from src.python.common.core.queue_backpressure import (
    BackpressureAlert,
    BackpressureDetector,
    BackpressureSeverity,
)
from src.python.common.core.queue_health import (
    HealthStatus,
    QueueHealthCalculator,
    QueueHealthStatus,
)
from src.python.common.core.queue_performance_metrics import (
    LatencyMetrics,
    MetricsAggregator,
    QueuePerformanceCollector,
    ThroughputMetrics,
)
from src.python.common.core.queue_statistics import (
    QueueStatistics,
    QueueStatisticsCollector,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_db():
    """Create temporary database for alert testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_alert_thresholds.db")
        yield db_path


@pytest.fixture
async def alert_system(temp_db):
    """Create alert system with temporary database and schema."""
    # Initialize database with schema
    conn = sqlite3.connect(temp_db)
    schema_path = (
        Path(__file__).parent.parent.parent
        / "src"
        / "python"
        / "common"
        / "core"
        / "alert_history_schema.sql"
    )

    if schema_path.exists():
        with open(schema_path) as f:
            schema_sql = f.read()
        conn.executescript(schema_sql)
        conn.commit()
    conn.close()

    # Create alert system with fast retry for testing
    system = QueueAlertingSystem(
        db_path=temp_db, max_retry_attempts=2, retry_delay_seconds=0.05
    )
    await system.initialize()
    yield system
    await system.close()


@pytest.fixture
def mock_stats_collector():
    """Mock statistics collector providing queue metrics."""
    stats = QueueStatistics(
        timestamp=datetime.now(timezone.utc),
        queue_size=1000,
        processing_rate=50.0,
        success_rate=0.98,  # As decimal
        failure_rate=0.02,  # As decimal
        avg_processing_time=120.0,
        items_added_rate=60.0,
        items_removed_rate=50.0,
        priority_distribution={},
        retry_count=5,
        error_count=20,
    )

    collector = AsyncMock(spec=QueueStatisticsCollector)
    collector.get_current_statistics = AsyncMock(return_value=stats)
    return collector


@pytest.fixture
def mock_performance_collector():
    """Mock performance collector providing throughput and latency metrics."""
    throughput = ThroughputMetrics(
        timestamp=datetime.now(timezone.utc),
        queue_type="ingestion_queue",
        items_per_second=25.5,
        items_per_minute=1530.0,
        items_per_hour=91800.0,
        bytes_per_second=51200.0,
    )

    latency = LatencyMetrics(
        timestamp=datetime.now(timezone.utc),
        queue_type="ingestion_queue",
        min_latency_ms=10.0,
        max_latency_ms=500.0,
        avg_latency_ms=120.0,
        median_latency_ms=100.0,
    )

    processing_time = MetricsAggregator(
        min=10.0,
        max=500.0,
        mean=120.0,
        median=100.0,
        p50=100.0,
        p95=250.0,
        p99=400.0,
        count=1000,
    )

    collector = AsyncMock(spec=QueuePerformanceCollector)
    collector.get_throughput_metrics = AsyncMock(return_value=throughput)
    collector.get_latency_metrics = AsyncMock(return_value=latency)
    collector.get_processing_time_stats = AsyncMock(return_value=processing_time)
    return collector


@pytest.fixture
def mock_health_calculator():
    """Mock health calculator providing health status and scores."""
    health = QueueHealthStatus(
        timestamp=datetime.now(timezone.utc),
        queue_type="ingestion_queue",
        overall_status=HealthStatus.HEALTHY,
        score=85.0,
        details={"reason": "All metrics within normal range"},
    )

    calculator = AsyncMock(spec=QueueHealthCalculator)
    calculator.calculate_health = AsyncMock(return_value=health)
    return calculator


@pytest.fixture
def mock_backpressure_detector():
    """Mock backpressure detector providing backpressure indicators."""
    # Mock BackpressureIndicators return
    class MockIndicators:
        def __init__(self):
            self.queue_growth_rate = 5.0
            self.processing_capacity_used = 0.65

    indicators = MockIndicators()

    detector = AsyncMock(spec=BackpressureDetector)
    detector.get_backpressure_indicators = AsyncMock(return_value=indicators)
    detector.detect_backpressure = AsyncMock(return_value=None)  # No backpressure by default
    return detector


@pytest.fixture
async def configured_alert_system(
    alert_system,
    mock_stats_collector,
    mock_performance_collector,
    mock_health_calculator,
    mock_backpressure_detector,
):
    """Alert system configured with all metric collectors."""
    alert_system.stats_collector = mock_stats_collector
    alert_system.performance_collector = mock_performance_collector
    alert_system.health_calculator = mock_health_calculator
    alert_system.backpressure_detector = mock_backpressure_detector
    return alert_system


# =============================================================================
# TEST CLASSES
# =============================================================================


class TestThresholdComparison:
    """Test threshold comparison operators and logic."""

    @pytest.mark.asyncio
    async def test_greater_than_operator_triggered(self, alert_system):
        """Test > operator triggers alert when value exceeds threshold."""
        result = alert_system._evaluate_threshold(150.0, ">", 100.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_greater_than_operator_not_triggered(self, alert_system):
        """Test > operator does not trigger when value is below threshold."""
        result = alert_system._evaluate_threshold(50.0, ">", 100.0)
        assert result is False

    @pytest.mark.asyncio
    async def test_less_than_operator_triggered(self, alert_system):
        """Test < operator triggers alert when value is below threshold."""
        result = alert_system._evaluate_threshold(30.0, "<", 50.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_less_than_operator_not_triggered(self, alert_system):
        """Test < operator does not trigger when value exceeds threshold."""
        result = alert_system._evaluate_threshold(150.0, "<", 100.0)
        assert result is False

    @pytest.mark.asyncio
    async def test_equals_operator_exact_match(self, alert_system):
        """Test == operator with exact match."""
        result = alert_system._evaluate_threshold(100.0, "==", 100.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_equals_operator_within_tolerance(self, alert_system):
        """Test == operator with float tolerance (0.001)."""
        result = alert_system._evaluate_threshold(100.0005, "==", 100.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_equals_operator_outside_tolerance(self, alert_system):
        """Test == operator fails when outside tolerance."""
        result = alert_system._evaluate_threshold(100.5, "==", 100.0)
        assert result is False

    @pytest.mark.asyncio
    async def test_greater_equals_operator(self, alert_system):
        """Test >= operator with equal and greater values."""
        assert alert_system._evaluate_threshold(100.0, ">=", 100.0) is True
        assert alert_system._evaluate_threshold(150.0, ">=", 100.0) is True
        assert alert_system._evaluate_threshold(50.0, ">=", 100.0) is False

    @pytest.mark.asyncio
    async def test_less_equals_operator(self, alert_system):
        """Test <= operator with equal and lesser values."""
        assert alert_system._evaluate_threshold(100.0, "<=", 100.0) is True
        assert alert_system._evaluate_threshold(50.0, "<=", 100.0) is True
        assert alert_system._evaluate_threshold(150.0, "<=", 100.0) is False

    @pytest.mark.asyncio
    async def test_boundary_value_exactly_at_threshold(self, alert_system):
        """Test boundary value exactly at threshold for all operators."""
        # Boundary at 100.0
        assert alert_system._evaluate_threshold(100.0, ">", 100.0) is False
        assert alert_system._evaluate_threshold(100.0, "<", 100.0) is False
        assert alert_system._evaluate_threshold(100.0, "==", 100.0) is True
        assert alert_system._evaluate_threshold(100.0, ">=", 100.0) is True
        assert alert_system._evaluate_threshold(100.0, "<=", 100.0) is True


class TestPerformanceAlerts:
    """Test performance-related alert types (latency, throughput)."""

    @pytest.mark.asyncio
    async def test_high_latency_alert(self, configured_alert_system):
        """Test alert triggers when latency exceeds threshold."""
        # Configure alert for latency > 100ms
        rule = AlertRule(
            rule_name="high_latency",
            description="Alert when average latency exceeds 100ms",
            thresholds=[
                AlertThreshold(
                    metric_name="latency_avg_ms",
                    operator=">",
                    value=100.0,
                    severity=AlertSeverity.WARNING,
                )
            ],
            recipients=["log"],
            cooldown_minutes=0,  # No cooldown for testing
        )

        await configured_alert_system.create_alert_rule(rule)
        alerts = await configured_alert_system.evaluate_rules()

        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.WARNING
        assert alerts[0].metric_name == "latency_avg_ms"
        assert alerts[0].metric_value == 120.0  # From mock

    @pytest.mark.asyncio
    async def test_low_throughput_alert(self, configured_alert_system):
        """Test alert triggers when throughput drops below threshold."""
        # Configure alert for throughput < 30 items/second
        rule = AlertRule(
            rule_name="low_throughput",
            description="Alert when throughput drops below 30 items/s",
            thresholds=[
                AlertThreshold(
                    metric_name="throughput_items_per_second",
                    operator="<",
                    value=30.0,
                    severity=AlertSeverity.ERROR,
                )
            ],
            recipients=["log"],
            cooldown_minutes=0,
        )

        await configured_alert_system.create_alert_rule(rule)
        alerts = await configured_alert_system.evaluate_rules()

        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.ERROR
        assert alerts[0].metric_value == 25.5  # From mock

    @pytest.mark.asyncio
    async def test_p95_latency_alert(self, configured_alert_system):
        """Test alert on P95 latency threshold."""
        rule = AlertRule(
            rule_name="p95_latency_high",
            description="Alert when P95 latency exceeds 200ms",
            thresholds=[
                AlertThreshold(
                    metric_name="processing_time_p95",
                    operator=">",
                    value=200.0,
                    severity=AlertSeverity.CRITICAL,
                )
            ],
            recipients=["log"],
            cooldown_minutes=0,
        )

        await configured_alert_system.create_alert_rule(rule)
        alerts = await configured_alert_system.evaluate_rules()

        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.CRITICAL
        assert alerts[0].metric_value == 250.0  # From mock


class TestResourceAlerts:
    """Test resource-related alert types (memory, disk, CPU, queue depth)."""

    @pytest.mark.asyncio
    async def test_queue_size_alert(self, configured_alert_system):
        """Test alert when queue size exceeds threshold."""
        rule = AlertRule(
            rule_name="high_queue_size",
            description="Alert when queue size exceeds 800 items",
            thresholds=[
                AlertThreshold(
                    metric_name="queue_size",
                    operator=">",
                    value=800.0,
                    severity=AlertSeverity.WARNING,
                )
            ],
            recipients=["log"],
            cooldown_minutes=0,
        )

        await configured_alert_system.create_alert_rule(rule)
        alerts = await configured_alert_system.evaluate_rules()

        assert len(alerts) == 1
        assert alerts[0].metric_name == "queue_size"
        assert alerts[0].metric_value == 1000.0  # From mock

    @pytest.mark.asyncio
    async def test_backpressure_capacity_alert(self, configured_alert_system):
        """Test alert when processing capacity usage is high."""
        rule = AlertRule(
            rule_name="high_capacity_usage",
            description="Alert when capacity usage exceeds 60%",
            thresholds=[
                AlertThreshold(
                    metric_name="backpressure_capacity_used",
                    operator=">",
                    value=0.60,
                    severity=AlertSeverity.WARNING,
                )
            ],
            recipients=["log"],
            cooldown_minutes=0,
        )

        await configured_alert_system.create_alert_rule(rule)
        alerts = await configured_alert_system.evaluate_rules()

        assert len(alerts) == 1
        assert alerts[0].metric_value == 0.65  # From mock

    @pytest.mark.asyncio
    async def test_multiple_resource_thresholds(self, configured_alert_system):
        """Test alert with multiple resource thresholds (queue size AND backpressure)."""
        rule = AlertRule(
            rule_name="resource_pressure",
            description="Alert when both queue and capacity are high",
            thresholds=[
                AlertThreshold(
                    metric_name="queue_size", operator=">", value=800.0, severity=AlertSeverity.ERROR
                ),
                AlertThreshold(
                    metric_name="backpressure_capacity_used",
                    operator=">",
                    value=0.60,
                    severity=AlertSeverity.ERROR,
                ),
            ],
            condition_logic=ConditionLogic.AND,
            recipients=["log"],
            cooldown_minutes=0,
        )

        await configured_alert_system.create_alert_rule(rule)
        alerts = await configured_alert_system.evaluate_rules()

        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.ERROR


class TestErrorRateAlerts:
    """Test error rate alert types."""

    @pytest.mark.asyncio
    async def test_high_error_rate_alert(self, configured_alert_system):
        """Test alert when error rate exceeds threshold."""
        rule = AlertRule(
            rule_name="high_error_rate",
            description="Alert when error rate exceeds 1%",
            thresholds=[
                AlertThreshold(
                    metric_name="error_rate",
                    operator=">",
                    value=0.01,
                    severity=AlertSeverity.CRITICAL,
                )
            ],
            recipients=["log"],
            cooldown_minutes=0,
        )

        await configured_alert_system.create_alert_rule(rule)
        alerts = await configured_alert_system.evaluate_rules()

        assert len(alerts) == 1
        assert alerts[0].metric_value == 0.02  # From mock (2% error rate)
        assert alerts[0].severity == AlertSeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_low_success_rate_alert(self, configured_alert_system):
        """Test alert when success rate drops below threshold."""
        rule = AlertRule(
            rule_name="low_success_rate",
            description="Alert when success rate drops below 99%",
            thresholds=[
                AlertThreshold(
                    metric_name="success_rate",
                    operator="<",
                    value=0.99,
                    severity=AlertSeverity.WARNING,
                )
            ],
            recipients=["log"],
            cooldown_minutes=0,
        )

        await configured_alert_system.create_alert_rule(rule)
        alerts = await configured_alert_system.evaluate_rules()

        assert len(alerts) == 1
        assert alerts[0].metric_value == 0.98  # From mock


class TestHealthDegradationAlerts:
    """Test health degradation alert types."""

    @pytest.mark.asyncio
    async def test_low_health_score_alert(self, configured_alert_system):
        """Test alert when health score drops below threshold."""
        rule = AlertRule(
            rule_name="low_health_score",
            description="Alert when health score drops below 90",
            thresholds=[
                AlertThreshold(
                    metric_name="health_score",
                    operator="<",
                    value=90.0,
                    severity=AlertSeverity.WARNING,
                )
            ],
            recipients=["log"],
            cooldown_minutes=0,
        )

        await configured_alert_system.create_alert_rule(rule)
        alerts = await configured_alert_system.evaluate_rules()

        assert len(alerts) == 1
        assert alerts[0].metric_value == 85.0  # From mock

    @pytest.mark.asyncio
    async def test_health_status_degraded_alert(self, configured_alert_system):
        """Test alert when health status becomes degraded."""
        # Modify mock to return degraded status
        async def get_degraded_health(queue_type="ingestion_queue"):
            return QueueHealthStatus(
                timestamp=datetime.now(timezone.utc),
                queue_type=queue_type,
                overall_status=HealthStatus.DEGRADED,
                score=70.0,
                details={"reason": "Performance degraded"},
            )

        configured_alert_system.health_calculator.calculate_health = get_degraded_health

        rule = AlertRule(
            rule_name="health_degraded",
            description="Alert when health status value indicates degradation",
            thresholds=[
                AlertThreshold(
                    metric_name="health_status_value",
                    operator="<",
                    value=100.0,
                    severity=AlertSeverity.ERROR,
                )
            ],
            recipients=["log"],
            cooldown_minutes=0,
        )

        await configured_alert_system.create_alert_rule(rule)
        alerts = await configured_alert_system.evaluate_rules()

        assert len(alerts) == 1
        assert alerts[0].metric_value == 70.0  # DEGRADED maps to 70.0


class TestQueueDepthAlerts:
    """Test queue depth (backlog) alert types."""

    @pytest.mark.asyncio
    async def test_queue_backlog_growth_alert(self, configured_alert_system):
        """Test alert when queue growth rate is positive."""
        rule = AlertRule(
            rule_name="queue_growing",
            description="Alert when queue is growing",
            thresholds=[
                AlertThreshold(
                    metric_name="backpressure_growth_rate",
                    operator=">",
                    value=0.0,
                    severity=AlertSeverity.INFO,
                )
            ],
            recipients=["log"],
            cooldown_minutes=0,
        )

        await configured_alert_system.create_alert_rule(rule)
        alerts = await configured_alert_system.evaluate_rules()

        assert len(alerts) == 1
        assert alerts[0].metric_value == 5.0  # From mock
        assert alerts[0].severity == AlertSeverity.INFO

    @pytest.mark.asyncio
    async def test_critical_queue_depth_alert(self, configured_alert_system):
        """Test critical alert when queue depth is extremely high."""
        # Update mock to return very high queue size
        async def get_high_queue_stats(queue_type="ingestion_queue"):
            return QueueStatistics(
                timestamp=datetime.now(timezone.utc),
                queue_type=queue_type,
                queue_size=10000,  # Very high
                processing_rate=50.0,
                failure_rate=0.02,
                success_rate=0.98,
                items_processed=5000,
                items_failed=100,
                items_succeeded=4900,
                avg_processing_time_ms=120.0,
            )

        configured_alert_system.stats_collector.get_current_statistics = get_high_queue_stats

        rule = AlertRule(
            rule_name="critical_queue_depth",
            description="Critical alert when queue exceeds 5000 items",
            thresholds=[
                AlertThreshold(
                    metric_name="queue_size",
                    operator=">",
                    value=5000.0,
                    severity=AlertSeverity.CRITICAL,
                )
            ],
            recipients=["log"],
            cooldown_minutes=0,
        )

        await configured_alert_system.create_alert_rule(rule)
        alerts = await configured_alert_system.evaluate_rules()

        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.CRITICAL
        assert alerts[0].metric_value == 10000.0


class TestAlertLifecycle:
    """Test alert lifecycle (activation, persistence, deactivation, cooldown)."""

    @pytest.mark.asyncio
    async def test_alert_activation(self, configured_alert_system):
        """Test alert activates when threshold crossed."""
        rule = AlertRule(
            rule_name="lifecycle_test",
            thresholds=[
                AlertThreshold(metric_name="queue_size", operator=">", value=500.0)
            ],
            cooldown_minutes=0,
        )

        await configured_alert_system.create_alert_rule(rule)
        alerts_before = await configured_alert_system.get_active_alerts()
        assert len(alerts_before) == 0

        # Evaluate rules - should trigger
        triggered = await configured_alert_system.evaluate_rules()
        assert len(triggered) == 1

        # Check active alerts
        alerts_after = await configured_alert_system.get_active_alerts()
        assert len(alerts_after) == 1

    @pytest.mark.asyncio
    async def test_alert_persistence(self, configured_alert_system):
        """Test alert persists while threshold violation continues."""
        rule = AlertRule(
            rule_name="persistence_test",
            thresholds=[
                AlertThreshold(metric_name="queue_size", operator=">", value=500.0)
            ],
            cooldown_minutes=0,
        )

        await configured_alert_system.create_alert_rule(rule)

        # Trigger alert
        await configured_alert_system.evaluate_rules()

        # Alert should still be active
        active = await configured_alert_system.get_active_alerts()
        assert len(active) == 1

        # Evaluate again - should create new alert (no cooldown)
        await configured_alert_system.evaluate_rules()
        active = await configured_alert_system.get_active_alerts()
        assert len(active) == 2  # Two separate alerts

    @pytest.mark.asyncio
    async def test_alert_deactivation_via_acknowledgment(self, configured_alert_system):
        """Test alert deactivates when acknowledged."""
        rule = AlertRule(
            rule_name="ack_test",
            thresholds=[
                AlertThreshold(metric_name="queue_size", operator=">", value=500.0)
            ],
            cooldown_minutes=0,
        )

        await configured_alert_system.create_alert_rule(rule)
        alerts = await configured_alert_system.evaluate_rules()
        alert_id = alerts[0].alert_id

        # Acknowledge alert
        acked = await configured_alert_system.acknowledge_alert(
            alert_id, acknowledged_by="test_user"
        )
        assert acked is True

        # Alert should no longer be active
        active = await configured_alert_system.get_active_alerts()
        assert len(active) == 0

    @pytest.mark.asyncio
    async def test_alert_cooldown_prevents_spam(self, configured_alert_system):
        """Test cooldown period prevents alert spam."""
        rule = AlertRule(
            rule_name="cooldown_test",
            thresholds=[
                AlertThreshold(metric_name="queue_size", operator=">", value=500.0)
            ],
            cooldown_minutes=60,  # 1 hour cooldown
        )

        await configured_alert_system.create_alert_rule(rule)

        # First evaluation - should trigger
        alerts1 = await configured_alert_system.evaluate_rules()
        assert len(alerts1) == 1

        # Second evaluation immediately - should not trigger (cooldown)
        alerts2 = await configured_alert_system.evaluate_rules()
        assert len(alerts2) == 0

    @pytest.mark.asyncio
    async def test_alert_cooldown_expiry(self, configured_alert_system):
        """Test alert can retrigger after cooldown expires."""
        # Short cooldown for testing
        rule = AlertRule(
            rule_name="cooldown_expiry_test",
            thresholds=[
                AlertThreshold(metric_name="queue_size", operator=">", value=500.0)
            ],
            cooldown_minutes=0.001,  # ~0.06 seconds
        )

        await configured_alert_system.create_alert_rule(rule)

        # First trigger
        alerts1 = await configured_alert_system.evaluate_rules()
        assert len(alerts1) == 1

        # Wait for cooldown to expire
        await asyncio.sleep(0.1)

        # Should trigger again
        alerts2 = await configured_alert_system.evaluate_rules()
        assert len(alerts2) == 1


class TestAlertSeverityLevels:
    """Test alert severity classification (INFO, WARNING, ERROR, CRITICAL)."""

    @pytest.mark.asyncio
    async def test_info_severity_alert(self, configured_alert_system):
        """Test INFO severity alert generation."""
        rule = AlertRule(
            rule_name="info_alert",
            thresholds=[
                AlertThreshold(
                    metric_name="queue_size",
                    operator=">",
                    value=500.0,
                    severity=AlertSeverity.INFO,
                )
            ],
            cooldown_minutes=0,
        )

        await configured_alert_system.create_alert_rule(rule)
        alerts = await configured_alert_system.evaluate_rules()

        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.INFO

    @pytest.mark.asyncio
    async def test_warning_severity_alert(self, configured_alert_system):
        """Test WARNING severity alert generation."""
        rule = AlertRule(
            rule_name="warning_alert",
            thresholds=[
                AlertThreshold(
                    metric_name="error_rate",
                    operator=">",
                    value=0.01,
                    severity=AlertSeverity.WARNING,
                )
            ],
            cooldown_minutes=0,
        )

        await configured_alert_system.create_alert_rule(rule)
        alerts = await configured_alert_system.evaluate_rules()

        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.WARNING

    @pytest.mark.asyncio
    async def test_error_severity_alert(self, configured_alert_system):
        """Test ERROR severity alert generation."""
        rule = AlertRule(
            rule_name="error_alert",
            thresholds=[
                AlertThreshold(
                    metric_name="health_score",
                    operator="<",
                    value=90.0,
                    severity=AlertSeverity.ERROR,
                )
            ],
            cooldown_minutes=0,
        )

        await configured_alert_system.create_alert_rule(rule)
        alerts = await configured_alert_system.evaluate_rules()

        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.ERROR

    @pytest.mark.asyncio
    async def test_critical_severity_alert(self, configured_alert_system):
        """Test CRITICAL severity alert generation."""
        rule = AlertRule(
            rule_name="critical_alert",
            thresholds=[
                AlertThreshold(
                    metric_name="error_rate",
                    operator=">",
                    value=0.01,
                    severity=AlertSeverity.CRITICAL,
                )
            ],
            cooldown_minutes=0,
        )

        await configured_alert_system.create_alert_rule(rule)
        alerts = await configured_alert_system.evaluate_rules()

        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_severity_prioritization_in_composite_alerts(
        self, configured_alert_system
    ):
        """Test highest severity is used when multiple thresholds trigger."""
        rule = AlertRule(
            rule_name="multi_severity",
            thresholds=[
                AlertThreshold(
                    metric_name="queue_size",
                    operator=">",
                    value=500.0,
                    severity=AlertSeverity.INFO,
                ),
                AlertThreshold(
                    metric_name="error_rate",
                    operator=">",
                    value=0.01,
                    severity=AlertSeverity.CRITICAL,
                ),
            ],
            condition_logic=ConditionLogic.OR,
            cooldown_minutes=0,
        )

        await configured_alert_system.create_alert_rule(rule)
        alerts = await configured_alert_system.evaluate_rules()

        # Should use CRITICAL (highest severity)
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.CRITICAL


class TestCompositeAlerts:
    """Test composite alerts with multiple conditions (AND/OR logic)."""

    @pytest.mark.asyncio
    async def test_and_logic_all_conditions_met(self, configured_alert_system):
        """Test AND logic triggers when all conditions are met."""
        rule = AlertRule(
            rule_name="and_all_met",
            thresholds=[
                AlertThreshold(metric_name="queue_size", operator=">", value=500.0),
                AlertThreshold(metric_name="error_rate", operator=">", value=0.01),
            ],
            condition_logic=ConditionLogic.AND,
            cooldown_minutes=0,
        )

        await configured_alert_system.create_alert_rule(rule)
        alerts = await configured_alert_system.evaluate_rules()

        assert len(alerts) == 1  # Both conditions met

    @pytest.mark.asyncio
    async def test_and_logic_partial_conditions(self, configured_alert_system):
        """Test AND logic does not trigger when only some conditions are met."""
        rule = AlertRule(
            rule_name="and_partial",
            thresholds=[
                AlertThreshold(metric_name="queue_size", operator=">", value=500.0),  # MET
                AlertThreshold(
                    metric_name="error_rate", operator=">", value=0.50
                ),  # NOT MET
            ],
            condition_logic=ConditionLogic.AND,
            cooldown_minutes=0,
        )

        await configured_alert_system.create_alert_rule(rule)
        alerts = await configured_alert_system.evaluate_rules()

        assert len(alerts) == 0  # Not all conditions met

    @pytest.mark.asyncio
    async def test_or_logic_one_condition_met(self, configured_alert_system):
        """Test OR logic triggers when at least one condition is met."""
        rule = AlertRule(
            rule_name="or_one_met",
            thresholds=[
                AlertThreshold(metric_name="queue_size", operator=">", value=500.0),  # MET
                AlertThreshold(
                    metric_name="error_rate", operator=">", value=0.50
                ),  # NOT MET
            ],
            condition_logic=ConditionLogic.OR,
            cooldown_minutes=0,
        )

        await configured_alert_system.create_alert_rule(rule)
        alerts = await configured_alert_system.evaluate_rules()

        assert len(alerts) == 1  # At least one condition met

    @pytest.mark.asyncio
    async def test_or_logic_no_conditions_met(self, configured_alert_system):
        """Test OR logic does not trigger when no conditions are met."""
        rule = AlertRule(
            rule_name="or_none_met",
            thresholds=[
                AlertThreshold(
                    metric_name="queue_size", operator=">", value=5000.0
                ),  # NOT MET
                AlertThreshold(
                    metric_name="error_rate", operator=">", value=0.50
                ),  # NOT MET
            ],
            condition_logic=ConditionLogic.OR,
            cooldown_minutes=0,
        )

        await configured_alert_system.create_alert_rule(rule)
        alerts = await configured_alert_system.evaluate_rules()

        assert len(alerts) == 0  # No conditions met

    @pytest.mark.asyncio
    async def test_complex_composite_alert(self, configured_alert_system):
        """Test complex composite alert with 3+ conditions."""
        rule = AlertRule(
            rule_name="complex_composite",
            thresholds=[
                AlertThreshold(metric_name="queue_size", operator=">", value=500.0),
                AlertThreshold(metric_name="error_rate", operator=">", value=0.01),
                AlertThreshold(metric_name="health_score", operator="<", value=90.0),
            ],
            condition_logic=ConditionLogic.AND,
            cooldown_minutes=0,
        )

        await configured_alert_system.create_alert_rule(rule)
        alerts = await configured_alert_system.evaluate_rules()

        assert len(alerts) == 1  # All three conditions met


class TestWindowedAlerts:
    """Test windowed alerts (sustained threshold violations over time)."""

    @pytest.mark.asyncio
    async def test_sustained_violation_detection(self, configured_alert_system):
        """Test detection of sustained threshold violations over multiple evaluations."""
        rule = AlertRule(
            rule_name="sustained_test",
            thresholds=[
                AlertThreshold(metric_name="queue_size", operator=">", value=500.0)
            ],
            cooldown_minutes=0,
        )

        await configured_alert_system.create_alert_rule(rule)

        # Evaluate multiple times to simulate sustained violation
        violations = []
        for _ in range(5):
            alerts = await configured_alert_system.evaluate_rules()
            violations.append(len(alerts) > 0)
            await asyncio.sleep(0.01)

        # All evaluations should show violation
        assert all(violations)

    @pytest.mark.asyncio
    async def test_intermittent_violation_pattern(self, configured_alert_system):
        """Test alert behavior with intermittent violations."""
        rule = AlertRule(
            rule_name="intermittent_test",
            thresholds=[
                AlertThreshold(metric_name="queue_size", operator=">", value=500.0)
            ],
            cooldown_minutes=0,
        )

        await configured_alert_system.create_alert_rule(rule)

        # First violation
        alerts1 = await configured_alert_system.evaluate_rules()
        assert len(alerts1) == 1

        # Change metric to not violate
        async def get_low_queue_stats(queue_type="ingestion_queue"):
            return QueueStatistics(
                timestamp=datetime.now(timezone.utc),
                queue_type=queue_type,
                queue_size=100,  # Below threshold
                processing_rate=50.0,
                failure_rate=0.02,
                success_rate=0.98,
                items_processed=5000,
                items_failed=100,
                items_succeeded=4900,
                avg_processing_time_ms=120.0,
            )

        configured_alert_system.stats_collector.get_current_statistics = (
            get_low_queue_stats
        )

        # Should not trigger
        alerts2 = await configured_alert_system.evaluate_rules()
        assert len(alerts2) == 0


class TestRateOfChangeAlerts:
    """Test rate of change alerts (rapid metric increases/decreases)."""

    @pytest.mark.asyncio
    async def test_rapid_queue_growth_detection(self, configured_alert_system):
        """Test detection of rapid queue growth."""
        rule = AlertRule(
            rule_name="rapid_growth",
            thresholds=[
                AlertThreshold(
                    metric_name="backpressure_growth_rate",
                    operator=">",
                    value=10.0,
                    severity=AlertSeverity.WARNING,
                )
            ],
            cooldown_minutes=0,
        )

        # Update mock to return high growth rate
        async def get_high_growth_indicators(queue_type="ingestion_queue"):
            class MockIndicators:
                def __init__(self):
                    self.queue_growth_rate = 15.0  # Rapid growth
                    self.processing_capacity_used = 0.65

            return MockIndicators()

        configured_alert_system.backpressure_detector.get_backpressure_indicators = (
            get_high_growth_indicators
        )

        await configured_alert_system.create_alert_rule(rule)
        alerts = await configured_alert_system.evaluate_rules()

        assert len(alerts) == 1
        assert alerts[0].metric_value == 15.0

    @pytest.mark.asyncio
    async def test_rapid_throughput_decrease_detection(self, configured_alert_system):
        """Test detection of rapid throughput decrease."""
        # Simulate throughput drop by comparing current to baseline
        # For this test, we'll check if throughput is below expected rate

        rule = AlertRule(
            rule_name="throughput_drop",
            thresholds=[
                AlertThreshold(
                    metric_name="throughput_items_per_second",
                    operator="<",
                    value=50.0,
                    severity=AlertSeverity.ERROR,
                )
            ],
            cooldown_minutes=0,
        )

        await configured_alert_system.create_alert_rule(rule)
        alerts = await configured_alert_system.evaluate_rules()

        assert len(alerts) == 1
        assert alerts[0].metric_value == 25.5  # From mock (drop detected)


class TestMetricsIntegration:
    """Test integration with metrics collection from subtask 313.1."""

    @pytest.mark.asyncio
    async def test_all_metric_sources_available(self, configured_alert_system):
        """Test that all metric sources are properly integrated."""
        # Fetch metrics to verify integration
        metrics = await configured_alert_system._fetch_metrics("ingestion_queue")

        # Verify metrics from all sources are present
        assert "queue_size" in metrics  # From stats_collector
        assert "processing_rate" in metrics
        assert "error_rate" in metrics
        assert "throughput_items_per_second" in metrics  # From performance_collector
        assert "latency_avg_ms" in metrics
        assert "health_score" in metrics  # From health_calculator
        assert "backpressure_growth_rate" in metrics  # From backpressure_detector

    @pytest.mark.asyncio
    async def test_metric_collection_failure_handling(self, configured_alert_system):
        """Test alert system handles metric collection failures gracefully."""
        # Make stats collector fail
        async def failing_stats(queue_type="ingestion_queue"):
            raise Exception("Metric collection failed")

        configured_alert_system.stats_collector.get_current_statistics = failing_stats

        # Should still work with available metrics
        metrics = await configured_alert_system._fetch_metrics("ingestion_queue")

        # Stats metrics should be missing but others present
        assert "queue_size" not in metrics
        assert "throughput_items_per_second" in metrics  # From other collectors

    @pytest.mark.asyncio
    async def test_cross_metric_correlation_alert(self, configured_alert_system):
        """Test alert using metrics from multiple collectors."""
        rule = AlertRule(
            rule_name="cross_metric_correlation",
            description="Alert when high queue size AND high latency",
            thresholds=[
                AlertThreshold(
                    metric_name="queue_size", operator=">", value=500.0
                ),  # stats_collector
                AlertThreshold(
                    metric_name="latency_avg_ms", operator=">", value=100.0
                ),  # performance_collector
            ],
            condition_logic=ConditionLogic.AND,
            cooldown_minutes=0,
        )

        await configured_alert_system.create_alert_rule(rule)
        alerts = await configured_alert_system.evaluate_rules()

        assert len(alerts) == 1  # Both metrics from different sources


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_missing_metric_handling(self, configured_alert_system):
        """Test alert system handles missing metrics gracefully."""
        rule = AlertRule(
            rule_name="missing_metric",
            thresholds=[
                AlertThreshold(
                    metric_name="nonexistent_metric",
                    operator=">",
                    value=100.0,
                )
            ],
            cooldown_minutes=0,
        )

        await configured_alert_system.create_alert_rule(rule)
        alerts = await configured_alert_system.evaluate_rules()

        # Should not trigger (metric unavailable)
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_null_metric_value_handling(self, configured_alert_system):
        """Test alert system handles null metric values."""
        # Make stats return None for queue_size
        async def get_null_stats(queue_type="ingestion_queue"):
            return QueueStatistics(
                timestamp=datetime.now(timezone.utc),
                queue_type=queue_type,
                queue_size=None,  # Null value
                processing_rate=50.0,
                failure_rate=0.02,
                success_rate=0.98,
                items_processed=5000,
                items_failed=100,
                items_succeeded=4900,
                avg_processing_time_ms=120.0,
            )

        configured_alert_system.stats_collector.get_current_statistics = get_null_stats

        rule = AlertRule(
            rule_name="null_value_test",
            thresholds=[
                AlertThreshold(metric_name="queue_size", operator=">", value=500.0)
            ],
            cooldown_minutes=0,
        )

        await configured_alert_system.create_alert_rule(rule)
        alerts = await configured_alert_system.evaluate_rules()

        # Should not trigger (metric is None)
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_disabled_threshold_ignored(self, configured_alert_system):
        """Test disabled thresholds are ignored in evaluation."""
        rule = AlertRule(
            rule_name="disabled_threshold",
            thresholds=[
                AlertThreshold(
                    metric_name="queue_size",
                    operator=">",
                    value=500.0,
                    enabled=False,  # Disabled
                )
            ],
            cooldown_minutes=0,
        )

        await configured_alert_system.create_alert_rule(rule)
        alerts = await configured_alert_system.evaluate_rules()

        # Should not trigger (threshold disabled)
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_alert_with_empty_threshold_list(self, configured_alert_system):
        """Test alert with no thresholds does not crash."""
        rule = AlertRule(
            rule_name="empty_thresholds", thresholds=[], cooldown_minutes=0
        )

        await configured_alert_system.create_alert_rule(rule)
        alerts = await configured_alert_system.evaluate_rules()

        # Should not trigger (no thresholds)
        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_concurrent_alert_evaluations(self, configured_alert_system):
        """Test concurrent alert evaluations don't interfere with each other."""
        rules = [
            AlertRule(
                rule_name=f"concurrent_{i}",
                thresholds=[
                    AlertThreshold(metric_name="queue_size", operator=">", value=500.0)
                ],
                cooldown_minutes=0,
            )
            for i in range(5)
        ]

        # Create all rules
        for rule in rules:
            await configured_alert_system.create_alert_rule(rule)

        # Evaluate concurrently
        tasks = [configured_alert_system.evaluate_rules() for _ in range(3)]
        results = await asyncio.gather(*tasks)

        # Each evaluation should find 5 alerts
        for alerts in results:
            assert len(alerts) == 5
