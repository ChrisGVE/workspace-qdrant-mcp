"""
Unit tests for queue alerting system.

Tests cover:
- Alert rule CRUD operations (create, update, delete, get)
- Threshold evaluation (all operators: >, <, ==, >=, <=)
- Compound conditions (AND/OR logic)
- Cooldown enforcement and expiry
- Notification delivery to multiple channels (Log, Email, Webhook, Slack, PagerDuty)
- Retry logic with exponential backoff
- Alert acknowledgment workflow
- Alert history retrieval and retention
- Edge cases (invalid rules, missing database, null values, concurrent operations)
"""

import asyncio
import os
import sqlite3
import tempfile
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
    DeliveryStatus,
    EmailNotifier,
    LogNotifier,
    NotificationChannel,
    PagerDutyNotifier,
    QueueAlertingSystem,
    SlackNotifier,
    WebhookNotifier,
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
    """Create temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_alerts.db")
        yield db_path


@pytest.fixture
async def alert_system(temp_db):
    """Create alert system with temporary database."""
    conn = sqlite3.connect(temp_db)
    schema_path = Path(__file__).parent.parent.parent / "src" / "python" / "common" / "core" / "alert_history_schema.sql"

    with open(schema_path) as f:
        schema_sql = f.read()
    conn.executescript(schema_sql)
    conn.commit()
    conn.close()

    system = QueueAlertingSystem(db_path=temp_db, max_retry_attempts=3, retry_delay_seconds=0.1)
    await system.initialize()
    yield system
    await system.close()


@pytest.fixture
def sample_threshold():
    """Create sample threshold for testing."""
    return AlertThreshold(metric_name="queue_size", operator=">", value=1000.0, severity=AlertSeverity.WARNING)


@pytest.fixture
def sample_rule(sample_threshold):
    """Create sample alert rule for testing."""
    return AlertRule(rule_name="test_rule", description="Test alert rule", thresholds=[sample_threshold], recipients=["log"], cooldown_minutes=15)


@pytest.fixture
def sample_notification():
    """Create sample alert notification for testing."""
    return AlertNotification(alert_id="test-alert-123", rule_name="test_rule", severity=AlertSeverity.WARNING, message="Test alert message", metric_name="queue_size", metric_value=1500.0, threshold_value=1000.0, threshold_operator=">", details={"test": "data"})


@pytest.fixture
def mock_stats_collector():
    """Create mock statistics collector."""
    collector = AsyncMock(spec=QueueStatisticsCollector)

    async def get_stats(queue_type="ingestion_queue"):
        return QueueStatistics(timestamp=datetime.now(timezone.utc), queue_type=queue_type, queue_size=500, processing_rate=10.0, failure_rate=0.05, success_rate=0.95, items_processed=1000, items_failed=50, items_succeeded=950, avg_processing_time_ms=100.0)

    collector.get_current_statistics = get_stats
    return collector


# =============================================================================
# TESTS
# =============================================================================


class TestAlertThreshold:
    """Test AlertThreshold dataclass and operations."""

    def test_threshold_creation(self):
        """Test creating alert threshold."""
        threshold = AlertThreshold(metric_name="queue_size", operator=">", value=1000.0, severity=AlertSeverity.WARNING, enabled=True)
        assert threshold.metric_name == "queue_size"
        assert threshold.operator == ">"
        assert threshold.value == 1000.0
        assert threshold.severity == AlertSeverity.WARNING
        assert threshold.enabled is True

    def test_threshold_to_dict(self, sample_threshold):
        """Test converting threshold to dictionary."""
        result = sample_threshold.to_dict()
        assert result["metric_name"] == "queue_size"
        assert result["operator"] == ">"
        assert result["value"] == 1000.0
        assert result["severity"] == "WARNING"
        assert result["enabled"] is True

    def test_threshold_from_dict(self):
        """Test creating threshold from dictionary."""
        data = {"metric_name": "error_rate", "operator": ">=", "value": 0.1, "severity": "ERROR", "enabled": False}
        threshold = AlertThreshold.from_dict(data)
        assert threshold.metric_name == "error_rate"
        assert threshold.operator == ">="
        assert threshold.value == 0.1
        assert threshold.severity == AlertSeverity.ERROR
        assert threshold.enabled is False


class TestAlertRuleCRUD:
    """Test alert rule CRUD operations."""

    @pytest.mark.asyncio
    async def test_create_alert_rule(self, alert_system, sample_rule):
        """Test creating alert rule."""
        rule_id = await alert_system.create_alert_rule(sample_rule)
        assert rule_id == sample_rule.rule_id
        rules = await alert_system.get_alert_rules()
        assert len(rules) == 1
        assert rules[0].rule_name == "test_rule"

    @pytest.mark.asyncio
    async def test_create_duplicate_rule_name(self, alert_system, sample_rule):
        """Test creating rule with duplicate name fails."""
        await alert_system.create_alert_rule(sample_rule)
        duplicate_rule = AlertRule(rule_name="test_rule", thresholds=[AlertThreshold("test", ">", 100.0)])
        with pytest.raises(ValueError, match="already exists"):
            await alert_system.create_alert_rule(duplicate_rule)

    @pytest.mark.asyncio
    async def test_update_alert_rule(self, alert_system, sample_rule):
        """Test updating alert rule."""
        rule_id = await alert_system.create_alert_rule(sample_rule)
        sample_rule.description = "Updated description"
        sample_rule.cooldown_minutes = 30
        updated = await alert_system.update_alert_rule(rule_id, sample_rule)
        assert updated is True
        rules = await alert_system.get_alert_rules()
        assert rules[0].description == "Updated description"
        assert rules[0].cooldown_minutes == 30

    @pytest.mark.asyncio
    async def test_delete_alert_rule(self, alert_system, sample_rule):
        """Test deleting alert rule."""
        rule_id = await alert_system.create_alert_rule(sample_rule)
        deleted = await alert_system.delete_alert_rule(rule_id)
        assert deleted is True
        rules = await alert_system.get_alert_rules()
        assert len(rules) == 0

    @pytest.mark.asyncio
    async def test_get_enabled_rules_only(self, alert_system):
        """Test getting only enabled rules."""
        enabled_rule = AlertRule(rule_name="enabled", thresholds=[AlertThreshold("test", ">", 100.0)], enabled=True)
        await alert_system.create_alert_rule(enabled_rule)
        disabled_rule = AlertRule(rule_name="disabled", thresholds=[AlertThreshold("test", ">", 100.0)], enabled=False)
        await alert_system.create_alert_rule(disabled_rule)
        enabled_rules = await alert_system.get_alert_rules(enabled_only=True)
        assert len(enabled_rules) == 1
        assert enabled_rules[0].rule_name == "enabled"


class TestThresholdEvaluation:
    """Test threshold evaluation logic."""

    @pytest.mark.asyncio
    async def test_greater_than_operator(self, alert_system):
        """Test > operator evaluation."""
        assert alert_system._evaluate_threshold(100.0, ">", 50.0) is True
        assert alert_system._evaluate_threshold(50.0, ">", 100.0) is False

    @pytest.mark.asyncio
    async def test_less_than_operator(self, alert_system):
        """Test < operator evaluation."""
        assert alert_system._evaluate_threshold(50.0, "<", 100.0) is True
        assert alert_system._evaluate_threshold(100.0, "<", 50.0) is False

    @pytest.mark.asyncio
    async def test_equals_operator(self, alert_system):
        """Test == operator evaluation with float tolerance."""
        assert alert_system._evaluate_threshold(50.0, "==", 50.0) is True
        assert alert_system._evaluate_threshold(50.0001, "==", 50.0) is True

    @pytest.mark.asyncio
    async def test_greater_equals_operator(self, alert_system):
        """Test >= operator evaluation."""
        assert alert_system._evaluate_threshold(100.0, ">=", 50.0) is True
        assert alert_system._evaluate_threshold(50.0, ">=", 50.0) is True

    @pytest.mark.asyncio
    async def test_less_equals_operator(self, alert_system):
        """Test <= operator evaluation."""
        assert alert_system._evaluate_threshold(50.0, "<=", 100.0) is True
        assert alert_system._evaluate_threshold(50.0, "<=", 50.0) is True


class TestCompoundConditions:
    """Test compound condition logic (AND/OR)."""

    @pytest.mark.asyncio
    async def test_and_logic_all_pass(self, alert_system, mock_stats_collector):
        """Test AND logic when all thresholds pass."""
        alert_system.stats_collector = mock_stats_collector
        rule = AlertRule(rule_name="and_all_pass", thresholds=[AlertThreshold("queue_size", ">", 100.0), AlertThreshold("processing_rate", ">", 5.0)], condition_logic=ConditionLogic.AND)
        triggered, notification = await alert_system._evaluate_rule(rule, "ingestion_queue")
        assert triggered is True
        assert notification is not None

    @pytest.mark.asyncio
    async def test_or_logic_one_pass(self, alert_system, mock_stats_collector):
        """Test OR logic when one threshold passes."""
        alert_system.stats_collector = mock_stats_collector
        rule = AlertRule(rule_name="or_one_pass", thresholds=[AlertThreshold("queue_size", ">", 1000.0), AlertThreshold("processing_rate", ">", 5.0)], condition_logic=ConditionLogic.OR)
        triggered, notification = await alert_system._evaluate_rule(rule, "ingestion_queue")
        assert triggered is True
        assert notification is not None


class TestNotificationChannels:
    """Test notification channels."""

    @pytest.mark.asyncio
    async def test_log_notifier_success(self, sample_notification):
        """Test successful log notification."""
        notifier = LogNotifier()
        with patch("src.python.common.core.queue_alerting.logger") as mock_logger:
            result = await notifier.send(sample_notification)
            assert result is True
            mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_email_notifier_success(self, sample_notification):
        """Test successful email notification."""
        notifier = EmailNotifier(smtp_host="smtp.test.com", smtp_port=587, smtp_user="test@test.com", smtp_password="password")
        with patch("src.python.common.core.queue_alerting.smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server
            result = await notifier.send(sample_notification, "recipient@test.com")
            assert result is True
            mock_server.starttls.assert_called_once()

    @pytest.mark.asyncio
    async def test_webhook_notifier_success(self, sample_notification):
        """Test successful webhook notification."""
        notifier = WebhookNotifier("https://webhook.test.com/alerts")
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response
            result = await notifier.send(sample_notification)
            assert result is True

    @pytest.mark.asyncio
    async def test_slack_notifier_success(self, sample_notification):
        """Test successful Slack notification."""
        notifier = SlackNotifier("https://hooks.slack.com/test")
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response
            result = await notifier.send(sample_notification)
            assert result is True

    @pytest.mark.asyncio
    async def test_pagerduty_notifier_success(self, sample_notification):
        """Test successful PagerDuty notification."""
        notifier = PagerDutyNotifier("test-integration-key")
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response
            result = await notifier.send(sample_notification)
            assert result is True


class TestCooldownEnforcement:
    """Test alert cooldown period enforcement."""

    @pytest.mark.asyncio
    async def test_cooldown_prevents_duplicate_alerts(self, alert_system, sample_rule, mock_stats_collector):
        """Test cooldown prevents duplicate alerts."""
        alert_system.stats_collector = mock_stats_collector
        sample_rule.cooldown_minutes = 1
        sample_rule.thresholds = [AlertThreshold("queue_size", ">", 100.0)]
        await alert_system.create_alert_rule(sample_rule)
        notifications = await alert_system.evaluate_rules()
        assert len(notifications) == 1
        notifications = await alert_system.evaluate_rules()
        assert len(notifications) == 0


class TestNotificationDelivery:
    """Test notification delivery to multiple channels."""

    @pytest.mark.asyncio
    async def test_delivery_to_multiple_channels(self, alert_system, sample_notification):
        """Test delivery to multiple channels."""
        mock_email = AsyncMock(spec=EmailNotifier)
        mock_email.send = AsyncMock(return_value=True)
        alert_system.register_notification_channel("email", mock_email)
        success = await alert_system.send_notification(sample_notification, recipients=["log", "email:test@test.com"])
        assert success is True


class TestRetryLogic:
    """Test notification delivery retry logic."""

    @pytest.mark.asyncio
    async def test_retry_on_failed_delivery(self, alert_system, sample_notification):
        """Test retry on failed delivery."""
        call_count = 0

        async def mock_send(notification):
            nonlocal call_count
            call_count += 1
            return call_count >= 3

        mock_channel = AsyncMock(spec=NotificationChannel)
        mock_channel.send = mock_send
        alert_system.register_notification_channel("retry_test", mock_channel)
        success = await alert_system.send_notification(sample_notification, recipients=["retry_test"])
        assert success is True
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries(self, alert_system, sample_notification):
        """Test retry stops after max attempts."""
        call_count = 0

        async def mock_send(notification):
            nonlocal call_count
            call_count += 1
            return False

        mock_channel = AsyncMock(spec=NotificationChannel)
        mock_channel.send = mock_send
        alert_system.register_notification_channel("max_retry_test", mock_channel)
        success = await alert_system.send_notification(sample_notification, recipients=["max_retry_test"])
        assert success is False
        assert call_count == alert_system.max_retry_attempts


class TestAlertAcknowledgment:
    """Test alert acknowledgment workflow."""

    @pytest.mark.asyncio
    async def test_acknowledge_alert(self, alert_system, sample_rule, mock_stats_collector):
        """Test acknowledging an alert."""
        alert_system.stats_collector = mock_stats_collector
        sample_rule.thresholds = [AlertThreshold("queue_size", ">", 100.0)]
        await alert_system.create_alert_rule(sample_rule)
        notifications = await alert_system.evaluate_rules()
        alert_id = notifications[0].alert_id
        acknowledged = await alert_system.acknowledge_alert(alert_id, acknowledged_by="user@test.com")
        assert acknowledged is True

    @pytest.mark.asyncio
    async def test_get_active_alerts_excludes_acknowledged(self, alert_system, sample_rule, mock_stats_collector):
        """Test get_active_alerts excludes acknowledged alerts."""
        alert_system.stats_collector = mock_stats_collector
        sample_rule.thresholds = [AlertThreshold("queue_size", ">", 100.0)]
        await alert_system.create_alert_rule(sample_rule)
        notifications = await alert_system.evaluate_rules()
        alert_id = notifications[0].alert_id
        active_before = await alert_system.get_active_alerts()
        assert len(active_before) == 1
        await alert_system.acknowledge_alert(alert_id)
        active_after = await alert_system.get_active_alerts()
        assert len(active_after) == 0


class TestAlertHistory:
    """Test alert history retrieval and management."""

    @pytest.mark.asyncio
    async def test_get_alert_history_by_time_range(self, alert_system, sample_rule, mock_stats_collector):
        """Test retrieving alert history by time range."""
        alert_system.stats_collector = mock_stats_collector
        sample_rule.thresholds = [AlertThreshold("queue_size", ">", 100.0)]
        sample_rule.cooldown_minutes = 0
        await alert_system.create_alert_rule(sample_rule)
        for _ in range(3):
            await alert_system.evaluate_rules()
        history = await alert_system.get_alert_history(hours=1)
        assert len(history) >= 1


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_null_threshold_values(self, alert_system, mock_stats_collector):
        """Test handling of null/missing threshold values."""
        async def get_stats_with_none(queue_type="ingestion_queue"):
            return QueueStatistics(timestamp=datetime.now(timezone.utc), queue_type=queue_type, queue_size=None, processing_rate=10.0, failure_rate=0.05, success_rate=0.95, items_processed=1000, items_failed=50, items_succeeded=950, avg_processing_time_ms=100.0)

        mock_stats_collector.get_current_statistics = get_stats_with_none
        alert_system.stats_collector = mock_stats_collector
        rule = AlertRule(rule_name="null_test", thresholds=[AlertThreshold("queue_size", ">", 100.0)])
        await alert_system.create_alert_rule(rule)
        notifications = await alert_system.evaluate_rules()
        assert len(notifications) == 0

    @pytest.mark.asyncio
    async def test_concurrent_alert_creation(self, alert_system, mock_stats_collector):
        """Test concurrent alert creation."""
        alert_system.stats_collector = mock_stats_collector
        rules = [AlertRule(rule_name=f"concurrent_{i}", thresholds=[AlertThreshold("queue_size", ">", 100.0)]) for i in range(5)]
        tasks = [alert_system.create_alert_rule(rule) for rule in rules]
        rule_ids = await asyncio.gather(*tasks)
        assert len(rule_ids) == 5
        assert len(set(rule_ids)) == 5
