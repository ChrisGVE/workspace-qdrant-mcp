"""
Queue Alerting System

Provides comprehensive alerting capabilities for queue monitoring with threshold-based
alerts, multiple notification channels, cooldown periods, and delivery tracking.

Features:
    - Threshold-based alert rules with multiple conditions
    - Compound conditions (AND/OR logic)
    - Multiple notification channels (log, email, webhook, Slack, PagerDuty)
    - Cooldown periods to prevent alert spam
    - Alert acknowledgment workflow
    - Delivery tracking with retry logic
    - Integration with queue monitoring modules

Example:
    ```python
    from workspace_qdrant_mcp.core.queue_alerting import (
        QueueAlertingSystem,
        AlertRule,
        AlertThreshold,
        AlertSeverity,
    )

    # Initialize alerting system
    alerting = QueueAlertingSystem()
    await alerting.initialize()

    # Create alert rule
    rule = AlertRule(
        rule_name="high_queue_size",
        description="Alert when queue size exceeds 5000",
        thresholds=[
            AlertThreshold(
                metric_name="queue_size",
                operator=">",
                value=5000,
                severity=AlertSeverity.WARNING,
            )
        ],
        recipients=["email:ops@example.com", "log"],
        cooldown_minutes=15,
    )

    rule_id = await alerting.create_alert_rule(rule)

    # Evaluate all rules
    alerts = await alerting.evaluate_rules()

    # Get active alerts
    active = await alerting.get_active_alerts()

    # Acknowledge alert
    await alerting.acknowledge_alert(alert_id="alert-123", acknowledged_by="user@example.com")
    ```
"""

import asyncio
import json
import os
import smtplib
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from .queue_backpressure import BackpressureDetector, BackpressureSeverity
from .queue_health import HealthStatus, QueueHealthCalculator
from .queue_performance_metrics import QueuePerformanceCollector
from .queue_statistics import QueueStatisticsCollector


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ConditionLogic(str, Enum):
    """Condition combination logic for compound conditions."""

    AND = "AND"
    OR = "OR"


class DeliveryStatus(str, Enum):
    """Notification delivery status."""

    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class AlertThreshold:
    """
    Alert threshold definition.

    Attributes:
        metric_name: Name of metric to evaluate (e.g., "queue_size", "error_rate")
        operator: Comparison operator (">", "<", "==", ">=", "<=")
        value: Threshold value to compare against
        severity: Alert severity if threshold is exceeded
        enabled: Whether this threshold is active
    """

    metric_name: str
    operator: str
    value: float
    severity: AlertSeverity = AlertSeverity.WARNING
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "metric_name": self.metric_name,
            "operator": self.operator,
            "value": self.value,
            "severity": self.severity.value,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AlertThreshold":
        """Create from dictionary."""
        return cls(
            metric_name=data["metric_name"],
            operator=data["operator"],
            value=data["value"],
            severity=AlertSeverity(data.get("severity", "WARNING")),
            enabled=data.get("enabled", True),
        )


@dataclass
class AlertRule:
    """
    Alert rule configuration.

    Attributes:
        rule_name: Unique name for this rule
        description: Human-readable description
        thresholds: List of thresholds to evaluate
        condition_logic: How to combine thresholds (AND/OR)
        recipients: List of notification recipients
        cooldown_minutes: Minimum time between alerts for this rule
        enabled: Whether this rule is active
        rule_id: Unique identifier (auto-generated)
    """

    rule_name: str
    thresholds: list[AlertThreshold]
    recipients: list[str] = field(default_factory=list)
    description: str = ""
    condition_logic: ConditionLogic = ConditionLogic.AND
    cooldown_minutes: int = 15
    enabled: bool = True
    rule_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "description": self.description,
            "enabled": self.enabled,
            "condition_logic": self.condition_logic.value,
            "thresholds": [t.to_dict() for t in self.thresholds],
            "recipients": self.recipients,
            "cooldown_minutes": self.cooldown_minutes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AlertRule":
        """Create from dictionary."""
        return cls(
            rule_id=data["rule_id"],
            rule_name=data["rule_name"],
            description=data.get("description", ""),
            enabled=data.get("enabled", True),
            condition_logic=ConditionLogic(data.get("condition_logic", "AND")),
            thresholds=[AlertThreshold.from_dict(t) for t in data["thresholds"]],
            recipients=data.get("recipients", []),
            cooldown_minutes=data.get("cooldown_minutes", 15),
        )


@dataclass
class AlertNotification:
    """
    Alert notification.

    Attributes:
        alert_id: Unique identifier for this alert
        rule_name: Name of the rule that triggered this alert
        severity: Alert severity level
        message: Human-readable alert message
        metric_name: Primary metric that triggered the alert
        metric_value: Current value of the metric
        threshold_value: Threshold value that was exceeded
        threshold_operator: Comparison operator used
        details: Additional context and information
        timestamp: When alert was generated
        acknowledged: Whether alert has been acknowledged
        acknowledged_at: When alert was acknowledged
        acknowledged_by: Who acknowledged the alert
    """

    alert_id: str
    rule_name: str
    severity: AlertSeverity
    message: str
    metric_name: str | None = None
    metric_value: float | None = None
    threshold_value: float | None = None
    threshold_operator: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged: bool = False
    acknowledged_at: datetime | None = None
    acknowledged_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "alert_id": self.alert_id,
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "message": self.message,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold_value": self.threshold_value,
            "threshold_operator": self.threshold_operator,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
        }


class NotificationChannel:
    """Base class for notification channels."""

    async def send(self, notification: AlertNotification) -> bool:
        """
        Send notification via this channel.

        Args:
            notification: Alert notification to send

        Returns:
            True if sent successfully, False otherwise
        """
        raise NotImplementedError


class LogNotifier(NotificationChannel):
    """Log-based notification channel (always available)."""

    async def send(self, notification: AlertNotification) -> bool:
        """Log notification to loguru."""
        try:
            severity_map = {
                AlertSeverity.INFO: logger.info,
                AlertSeverity.WARNING: logger.warning,
                AlertSeverity.ERROR: logger.error,
                AlertSeverity.CRITICAL: logger.critical,
            }

            log_func = severity_map.get(notification.severity, logger.warning)
            log_func(
                f"ALERT [{notification.severity.value}] {notification.rule_name}: {notification.message}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to log alert notification: {e}")
            return False


class EmailNotifier(NotificationChannel):
    """Email notification channel via SMTP."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int = 587,
        smtp_user: str | None = None,
        smtp_password: str | None = None,
        from_addr: str = "alerts@workspace-qdrant.local",
        use_tls: bool = True,
    ):
        """
        Initialize email notifier.

        Args:
            smtp_host: SMTP server hostname
            smtp_port: SMTP server port
            smtp_user: SMTP authentication username
            smtp_password: SMTP authentication password
            from_addr: Sender email address
            use_tls: Whether to use TLS
        """
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.from_addr = from_addr
        self.use_tls = use_tls

    async def send(self, notification: AlertNotification, to_addr: str) -> bool:
        """
        Send email notification.

        Args:
            notification: Alert notification
            to_addr: Recipient email address

        Returns:
            True if sent successfully
        """
        try:
            # Create email message
            msg = MIMEText(self._format_message(notification), "plain")
            msg["Subject"] = f"[{notification.severity.value}] {notification.rule_name}"
            msg["From"] = self.from_addr
            msg["To"] = to_addr

            # Send via SMTP
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                if self.smtp_user and self.smtp_password:
                    server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

            logger.info(f"Sent email alert to {to_addr}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email alert to {to_addr}: {e}")
            return False

    def _format_message(self, notification: AlertNotification) -> str:
        """Format alert notification as email body."""
        lines = [
            f"Alert: {notification.rule_name}",
            f"Severity: {notification.severity.value}",
            f"Timestamp: {notification.timestamp.isoformat()}",
            "",
            f"Message: {notification.message}",
        ]

        if notification.metric_name:
            lines.extend([
                "",
                f"Metric: {notification.metric_name}",
                f"Current Value: {notification.metric_value}",
                f"Threshold: {notification.threshold_operator} {notification.threshold_value}",
            ])

        if notification.details:
            lines.extend(["", "Details:", json.dumps(notification.details, indent=2)])

        return "\n".join(lines)


class WebhookNotifier(NotificationChannel):
    """Webhook notification channel via HTTP POST."""

    def __init__(self, webhook_url: str, timeout: int = 10):
        """
        Initialize webhook notifier.

        Args:
            webhook_url: URL to POST notifications to
            timeout: Request timeout in seconds
        """
        self.webhook_url = webhook_url
        self.timeout = timeout

    async def send(self, notification: AlertNotification) -> bool:
        """
        Send webhook notification.

        Args:
            notification: Alert notification

        Returns:
            True if sent successfully
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.webhook_url,
                    json=notification.to_dict(),
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()

            logger.info(f"Sent webhook alert to {self.webhook_url}")
            return True

        except Exception as e:
            logger.error(f"Failed to send webhook alert to {self.webhook_url}: {e}")
            return False


class SlackNotifier(NotificationChannel):
    """Slack notification channel (requires slack_sdk)."""

    def __init__(self, webhook_url: str):
        """
        Initialize Slack notifier.

        Args:
            webhook_url: Slack webhook URL
        """
        self.webhook_url = webhook_url

    async def send(self, notification: AlertNotification) -> bool:
        """
        Send Slack notification.

        Args:
            notification: Alert notification

        Returns:
            True if sent successfully
        """
        try:
            # Use webhook for simplicity (no SDK required)
            color_map = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ffcc00",
                AlertSeverity.ERROR: "#ff0000",
                AlertSeverity.CRITICAL: "#990000",
            }

            payload = {
                "attachments": [
                    {
                        "color": color_map.get(notification.severity, "#808080"),
                        "title": f"[{notification.severity.value}] {notification.rule_name}",
                        "text": notification.message,
                        "fields": [
                            {
                                "title": "Metric",
                                "value": notification.metric_name or "N/A",
                                "short": True,
                            },
                            {
                                "title": "Value",
                                "value": str(notification.metric_value) if notification.metric_value is not None else "N/A",
                                "short": True,
                            },
                        ],
                        "ts": int(notification.timestamp.timestamp()),
                    }
                ]
            }

            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(self.webhook_url, json=payload)
                response.raise_for_status()

            logger.info("Sent Slack alert")
            return True

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False


class PagerDutyNotifier(NotificationChannel):
    """PagerDuty notification channel."""

    def __init__(self, integration_key: str):
        """
        Initialize PagerDuty notifier.

        Args:
            integration_key: PagerDuty integration key
        """
        self.integration_key = integration_key
        self.api_url = "https://events.pagerduty.com/v2/enqueue"

    async def send(self, notification: AlertNotification) -> bool:
        """
        Send PagerDuty notification.

        Args:
            notification: Alert notification

        Returns:
            True if sent successfully
        """
        try:
            severity_map = {
                AlertSeverity.INFO: "info",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.ERROR: "error",
                AlertSeverity.CRITICAL: "critical",
            }

            payload = {
                "routing_key": self.integration_key,
                "event_action": "trigger",
                "dedup_key": notification.alert_id,
                "payload": {
                    "summary": notification.message,
                    "severity": severity_map.get(notification.severity, "warning"),
                    "source": "workspace-qdrant-mcp",
                    "timestamp": notification.timestamp.isoformat(),
                    "custom_details": {
                        "rule_name": notification.rule_name,
                        "metric_name": notification.metric_name,
                        "metric_value": notification.metric_value,
                        "threshold_value": notification.threshold_value,
                        **notification.details,
                    },
                },
            }

            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(self.api_url, json=payload)
                response.raise_for_status()

            logger.info("Sent PagerDuty alert")
            return True

        except Exception as e:
            logger.error(f"Failed to send PagerDuty alert: {e}")
            return False


class QueueAlertingSystem:
    """
    Queue alerting system with threshold-based alerts and multiple notification channels.

    Manages alert rules, evaluates metrics against thresholds, and sends notifications
    via configured channels with retry logic and delivery tracking.
    """

    def __init__(
        self,
        db_path: str | None = None,
        stats_collector: QueueStatisticsCollector | None = None,
        performance_collector: QueuePerformanceCollector | None = None,
        health_calculator: QueueHealthCalculator | None = None,
        backpressure_detector: BackpressureDetector | None = None,
        max_retry_attempts: int = 3,
        retry_delay_seconds: int = 30,
    ):
        """
        Initialize queue alerting system.

        Args:
            db_path: Optional database path (defaults to ~/.workspace-qdrant/state.db)
            stats_collector: Optional statistics collector for metrics
            performance_collector: Optional performance collector for metrics
            health_calculator: Optional health calculator for metrics
            backpressure_detector: Optional backpressure detector for metrics
            max_retry_attempts: Maximum notification delivery retry attempts
            retry_delay_seconds: Delay between retry attempts
        """
        if db_path is None:
            db_path = str(Path.home() / ".workspace-qdrant" / "state.db")

        self.db_path = db_path
        self.stats_collector = stats_collector
        self.performance_collector = performance_collector
        self.health_calculator = health_calculator
        self.backpressure_detector = backpressure_detector
        self.max_retry_attempts = max_retry_attempts
        self.retry_delay_seconds = retry_delay_seconds

        self._initialized = False
        self._db: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()

        # Notification channels
        self._channels: dict[str, NotificationChannel] = {
            "log": LogNotifier(),
        }

    async def initialize(self):
        """Initialize the alerting system."""
        if self._initialized:
            return

        loop = asyncio.get_event_loop()

        # Ensure database directory exists
        db_dir = os.path.dirname(self.db_path)
        await loop.run_in_executor(None, lambda: os.makedirs(db_dir, exist_ok=True))

        # Open database connection (sync operation)
        def _open_connection():
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn

        self._db = await loop.run_in_executor(None, _open_connection)

        # Load schema (sync operation)
        schema_path = Path(__file__).parent / "alert_history_schema.sql"
        if schema_path.exists():
            def _load_schema():
                with open(schema_path) as f:
                    schema_sql = f.read()
                self._db.executescript(schema_sql)
                self._db.commit()

            await loop.run_in_executor(None, _load_schema)

        self._initialized = True
        logger.info(f"Queue alerting system initialized (db={self.db_path})")

    async def close(self):
        """Close the alerting system."""
        if not self._initialized:
            return

        if self._db:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._db.close)
            self._db = None

        self._initialized = False
        logger.info("Queue alerting system closed")

    def register_notification_channel(
        self, channel_name: str, channel: NotificationChannel
    ):
        """
        Register a notification channel.

        Args:
            channel_name: Unique name for the channel
            channel: Notification channel instance
        """
        self._channels[channel_name] = channel
        logger.info(f"Registered notification channel: {channel_name}")

    async def create_alert_rule(self, rule: AlertRule) -> str:
        """
        Create a new alert rule.

        Args:
            rule: Alert rule configuration

        Returns:
            Rule ID

        Raises:
            ValueError: If rule with same name already exists
        """
        async with self._lock:
            loop = asyncio.get_event_loop()

            # Check for duplicate name (sync operation)
            def _check_duplicate():
                cursor = self._db.execute(
                    "SELECT rule_id FROM alert_rules WHERE rule_name = ?",
                    (rule.rule_name,),
                )
                return cursor.fetchone()

            existing = await loop.run_in_executor(None, _check_duplicate)
            if existing:
                raise ValueError(f"Alert rule '{rule.rule_name}' already exists")

            # Insert rule (sync operation)
            def _insert_rule():
                now = time.time()
                self._db.execute(
                    """
                    INSERT INTO alert_rules (
                        rule_id, rule_name, description, enabled, condition_logic,
                        thresholds_json, recipients_json, cooldown_minutes,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rule.rule_id,
                        rule.rule_name,
                        rule.description,
                        rule.enabled,
                        rule.condition_logic.value,
                        json.dumps([t.to_dict() for t in rule.thresholds]),
                        json.dumps(rule.recipients),
                        rule.cooldown_minutes,
                        now,
                        now,
                    ),
                )
                self._db.commit()

            await loop.run_in_executor(None, _insert_rule)

        logger.info(f"Created alert rule: {rule.rule_name} (id={rule.rule_id})")
        return rule.rule_id

    async def update_alert_rule(self, rule_id: str, rule: AlertRule) -> bool:
        """
        Update an existing alert rule.

        Args:
            rule_id: Rule ID to update
            rule: Updated rule configuration

        Returns:
            True if updated, False if not found
        """
        async with self._lock:
            loop = asyncio.get_event_loop()

            def _update_rule():
                now = time.time()
                cursor = self._db.execute(
                    """
                    UPDATE alert_rules
                    SET rule_name = ?, description = ?, enabled = ?, condition_logic = ?,
                        thresholds_json = ?, recipients_json = ?, cooldown_minutes = ?,
                        updated_at = ?
                    WHERE rule_id = ?
                    """,
                    (
                        rule.rule_name,
                        rule.description,
                        rule.enabled,
                        rule.condition_logic.value,
                        json.dumps([t.to_dict() for t in rule.thresholds]),
                        json.dumps(rule.recipients),
                        rule.cooldown_minutes,
                        now,
                        rule_id,
                    ),
                )
                self._db.commit()
                return cursor.rowcount > 0

            updated = await loop.run_in_executor(None, _update_rule)

        if updated:
            logger.info(f"Updated alert rule: {rule_id}")
        else:
            logger.warning(f"Alert rule not found: {rule_id}")

        return updated

    async def delete_alert_rule(self, rule_id: str) -> bool:
        """
        Delete an alert rule.

        Args:
            rule_id: Rule ID to delete

        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            loop = asyncio.get_event_loop()

            def _delete_rule():
                cursor = self._db.execute(
                    "DELETE FROM alert_rules WHERE rule_id = ?", (rule_id,)
                )
                self._db.commit()
                return cursor.rowcount > 0

            deleted = await loop.run_in_executor(None, _delete_rule)

        if deleted:
            logger.info(f"Deleted alert rule: {rule_id}")
        else:
            logger.warning(f"Alert rule not found: {rule_id}")

        return deleted

    async def get_alert_rules(self, enabled_only: bool = False) -> list[AlertRule]:
        """
        Get all alert rules.

        Args:
            enabled_only: If True, return only enabled rules

        Returns:
            List of alert rules
        """
        loop = asyncio.get_event_loop()

        def _get_rules():
            query = "SELECT * FROM alert_rules"
            if enabled_only:
                query += " WHERE enabled = 1"

            cursor = self._db.execute(query)
            return cursor.fetchall()

        rows = await loop.run_in_executor(None, _get_rules)

        rules = []
        for row in rows:
            rule = AlertRule(
                rule_id=row["rule_id"],
                rule_name=row["rule_name"],
                description=row["description"] or "",
                enabled=bool(row["enabled"]),
                condition_logic=ConditionLogic(row["condition_logic"]),
                thresholds=[
                    AlertThreshold.from_dict(t)
                    for t in json.loads(row["thresholds_json"])
                ],
                recipients=json.loads(row["recipients_json"]),
                cooldown_minutes=row["cooldown_minutes"],
            )
            rules.append(rule)

        return rules

    async def evaluate_rules(self, queue_type: str = "ingestion_queue") -> list[AlertNotification]:
        """
        Evaluate all enabled alert rules and generate notifications.

        Args:
            queue_type: Queue type to evaluate

        Returns:
            List of triggered alert notifications
        """
        rules = await self.get_alert_rules(enabled_only=True)
        notifications = []

        for rule in rules:
            # Check cooldown
            if not await self._check_cooldown(rule):
                logger.debug(f"Rule {rule.rule_name} in cooldown period, skipping")
                continue

            # Evaluate rule
            triggered, notification = await self._evaluate_rule(rule, queue_type)

            if triggered and notification:
                # Record alert in history
                await self._record_alert(notification, rule.rule_id)

                # Update last triggered time
                await self._update_last_triggered(rule.rule_id)

                # Send notifications
                await self.send_notification(notification, rule.recipients)

                notifications.append(notification)

        return notifications

    async def send_notification(
        self,
        notification: AlertNotification,
        recipients: list[str] | None = None,
    ) -> bool:
        """
        Send notification to configured recipients.

        Args:
            notification: Alert notification to send
            recipients: Optional list of recipients (defaults to rule recipients)

        Returns:
            True if sent successfully to at least one channel
        """
        recipients = recipients or ["log"]
        success = False

        for recipient in recipients:
            # Parse recipient format: "channel:address" or just "channel"
            parts = recipient.split(":", 1)
            channel_name = parts[0]
            channel_address = parts[1] if len(parts) > 1 else None

            # Get channel
            channel = self._channels.get(channel_name)
            if not channel:
                logger.warning(f"Unknown notification channel: {channel_name}")
                continue

            # Send with retry
            delivery_success = await self._send_with_retry(
                notification, channel, channel_address
            )

            if delivery_success:
                success = True

        return success

    async def acknowledge_alert(
        self, alert_id: str, acknowledged_by: str | None = None
    ) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert ID to acknowledge
            acknowledged_by: Optional user/system that acknowledged

        Returns:
            True if acknowledged, False if not found
        """
        async with self._lock:
            loop = asyncio.get_event_loop()

            def _acknowledge():
                now = time.time()
                cursor = self._db.execute(
                    """
                    UPDATE alert_history
                    SET acknowledged = 1, acknowledged_at = ?, acknowledged_by = ?
                    WHERE alert_id = ?
                    """,
                    (now, acknowledged_by, alert_id),
                )
                self._db.commit()
                return cursor.rowcount > 0

            acknowledged = await loop.run_in_executor(None, _acknowledge)

        if acknowledged:
            logger.info(f"Acknowledged alert: {alert_id}")
        else:
            logger.warning(f"Alert not found: {alert_id}")

        return acknowledged

    async def get_active_alerts(self) -> list[AlertNotification]:
        """
        Get all active (unacknowledged) alerts.

        Returns:
            List of active alerts
        """
        loop = asyncio.get_event_loop()

        def _get_active():
            cursor = self._db.execute(
                """
                SELECT * FROM alert_history
                WHERE acknowledged = 0
                ORDER BY timestamp DESC
                """
            )
            return cursor.fetchall()

        rows = await loop.run_in_executor(None, _get_active)
        return [self._row_to_notification(row) for row in rows]

    async def get_alert_history(self, hours: int = 24) -> list[AlertNotification]:
        """
        Get alert history for specified time period.

        Args:
            hours: Number of hours of history to retrieve

        Returns:
            List of alert notifications
        """
        cutoff_time = time.time() - (hours * 3600)
        loop = asyncio.get_event_loop()

        def _get_history():
            cursor = self._db.execute(
                """
                SELECT * FROM alert_history
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
                """,
                (cutoff_time,),
            )
            return cursor.fetchall()

        rows = await loop.run_in_executor(None, _get_history)
        return [self._row_to_notification(row) for row in rows]

    async def _check_cooldown(self, rule: AlertRule) -> bool:
        """Check if rule is outside cooldown period."""
        loop = asyncio.get_event_loop()

        def _get_last_triggered():
            cursor = self._db.execute(
                "SELECT last_triggered_at FROM alert_rules WHERE rule_id = ?",
                (rule.rule_id,),
            )
            return cursor.fetchone()

        row = await loop.run_in_executor(None, _get_last_triggered)

        if not row or row["last_triggered_at"] is None:
            return True

        last_triggered = row["last_triggered_at"]
        cooldown_seconds = rule.cooldown_minutes * 60
        elapsed = time.time() - last_triggered

        return elapsed >= cooldown_seconds

    async def _evaluate_rule(
        self, rule: AlertRule, queue_type: str
    ) -> tuple[bool, AlertNotification | None]:
        """Evaluate a single rule against current metrics."""
        # Fetch current metrics
        metrics = await self._fetch_metrics(queue_type)

        # Evaluate each threshold
        threshold_results = []
        for threshold in rule.thresholds:
            if not threshold.enabled:
                continue

            metric_value = metrics.get(threshold.metric_name)
            if metric_value is None:
                logger.debug(f"Metric {threshold.metric_name} not available, skipping")
                continue

            # Evaluate threshold
            triggered = self._evaluate_threshold(
                metric_value, threshold.operator, threshold.value
            )
            threshold_results.append((triggered, threshold, metric_value))

        if not threshold_results:
            return False, None

        # Apply condition logic
        if rule.condition_logic == ConditionLogic.AND:
            rule_triggered = all(result[0] for result in threshold_results)
        else:  # OR
            rule_triggered = any(result[0] for result in threshold_results)

        if not rule_triggered:
            return False, None

        # Find highest severity triggered threshold
        triggered_thresholds = [
            (threshold, value)
            for triggered, threshold, value in threshold_results
            if triggered
        ]

        if not triggered_thresholds:
            return False, None

        # Sort by severity (CRITICAL > ERROR > WARNING > INFO)
        severity_order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.ERROR: 1,
            AlertSeverity.WARNING: 2,
            AlertSeverity.INFO: 3,
        }
        triggered_thresholds.sort(key=lambda x: severity_order[x[0].severity])
        primary_threshold, primary_value = triggered_thresholds[0]

        # Create notification
        notification = AlertNotification(
            alert_id=str(uuid.uuid4()),
            rule_name=rule.rule_name,
            severity=primary_threshold.severity,
            message=self._format_alert_message(rule, triggered_thresholds),
            metric_name=primary_threshold.metric_name,
            metric_value=primary_value,
            threshold_value=primary_threshold.value,
            threshold_operator=primary_threshold.operator,
            details={
                "queue_type": queue_type,
                "triggered_thresholds": [
                    {
                        "metric_name": t.metric_name,
                        "value": v,
                        "threshold": t.value,
                        "operator": t.operator,
                    }
                    for t, v in triggered_thresholds
                ],
            },
        )

        return True, notification

    def _evaluate_threshold(self, value: float, operator: str, threshold: float) -> bool:
        """Evaluate a threshold condition."""
        if operator == ">":
            return value > threshold
        elif operator == "<":
            return value < threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "==":
            return abs(value - threshold) < 0.001  # Float comparison tolerance
        else:
            logger.warning(f"Unknown operator: {operator}")
            return False

    def _format_alert_message(
        self, rule: AlertRule, triggered_thresholds: list[tuple]
    ) -> str:
        """Format alert message from triggered thresholds."""
        if len(triggered_thresholds) == 1:
            threshold, value = triggered_thresholds[0]
            return (
                f"{threshold.metric_name} is {value:.2f} "
                f"(threshold: {threshold.operator} {threshold.value:.2f})"
            )
        else:
            metric_names = [t.metric_name for t, _ in triggered_thresholds]
            return f"Multiple thresholds triggered: {', '.join(metric_names)}"

    async def _fetch_metrics(self, queue_type: str) -> dict[str, float]:
        """Fetch current metrics from monitoring modules."""
        metrics = {}

        # Queue statistics
        if self.stats_collector:
            try:
                stats = await self.stats_collector.get_current_statistics(queue_type=queue_type)
                metrics.update({
                    "queue_size": float(stats.queue_size),
                    "processing_rate": stats.processing_rate,
                    "error_rate": stats.failure_rate,
                    "success_rate": stats.success_rate,
                })
            except Exception as e:
                logger.warning(f"Failed to fetch queue statistics: {e}")

        # Performance metrics
        if self.performance_collector:
            try:
                throughput = await self.performance_collector.get_throughput_metrics()
                latency = await self.performance_collector.get_latency_metrics()
                processing_time = await self.performance_collector.get_processing_time_stats()

                metrics.update({
                    "throughput_items_per_second": throughput.items_per_second,
                    "throughput_items_per_minute": throughput.items_per_minute,
                    "latency_avg_ms": latency.avg_latency_ms,
                    "latency_max_ms": latency.max_latency_ms,
                    "processing_time_p95": processing_time.p95,
                    "processing_time_p99": processing_time.p99,
                })
            except Exception as e:
                logger.warning(f"Failed to fetch performance metrics: {e}")

        # Health score
        if self.health_calculator:
            try:
                health = await self.health_calculator.calculate_health(queue_type=queue_type)
                metrics["health_score"] = health.score

                # Map health status to numeric value
                status_values = {
                    HealthStatus.HEALTHY: 100.0,
                    HealthStatus.DEGRADED: 70.0,
                    HealthStatus.UNHEALTHY: 40.0,
                    HealthStatus.CRITICAL: 0.0,
                }
                metrics["health_status_value"] = status_values.get(
                    health.overall_status, 50.0
                )
            except Exception as e:
                logger.warning(f"Failed to fetch health metrics: {e}")

        # Backpressure
        if self.backpressure_detector:
            try:
                indicators = await self.backpressure_detector.get_backpressure_indicators(
                    queue_type=queue_type
                )
                metrics.update({
                    "backpressure_growth_rate": indicators.queue_growth_rate,
                    "backpressure_capacity_used": indicators.processing_capacity_used,
                })

                # Map backpressure severity to numeric value
                severity_values = {
                    BackpressureSeverity.NONE: 0.0,
                    BackpressureSeverity.LOW: 25.0,
                    BackpressureSeverity.MEDIUM: 50.0,
                    BackpressureSeverity.HIGH: 75.0,
                    BackpressureSeverity.CRITICAL: 100.0,
                }

                alert = await self.backpressure_detector.detect_backpressure(queue_type=queue_type)
                if alert:
                    metrics["backpressure_severity_value"] = severity_values.get(
                        alert.severity, 0.0
                    )
                else:
                    metrics["backpressure_severity_value"] = 0.0

            except Exception as e:
                logger.warning(f"Failed to fetch backpressure metrics: {e}")

        return metrics

    async def _record_alert(self, notification: AlertNotification, rule_id: str):
        """Record alert in history."""
        loop = asyncio.get_event_loop()

        def _insert_alert():
            self._db.execute(
                """
                INSERT INTO alert_history (
                    alert_id, rule_id, rule_name, severity, message,
                    metric_name, metric_value, threshold_value, threshold_operator,
                    details_json, timestamp, acknowledged
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    notification.alert_id,
                    rule_id,
                    notification.rule_name,
                    notification.severity.value,
                    notification.message,
                    notification.metric_name,
                    notification.metric_value,
                    notification.threshold_value,
                    notification.threshold_operator,
                    json.dumps(notification.details),
                    notification.timestamp.timestamp(),
                    0,
                ),
            )
            self._db.commit()

        await loop.run_in_executor(None, _insert_alert)

    async def _update_last_triggered(self, rule_id: str):
        """Update last triggered timestamp for a rule."""
        loop = asyncio.get_event_loop()

        def _update():
            self._db.execute(
                "UPDATE alert_rules SET last_triggered_at = ? WHERE rule_id = ?",
                (time.time(), rule_id),
            )
            self._db.commit()

        await loop.run_in_executor(None, _update)

    async def _send_with_retry(
        self,
        notification: AlertNotification,
        channel: NotificationChannel,
        address: str | None = None,
    ) -> bool:
        """Send notification with retry logic."""
        delivery_id = str(uuid.uuid4())
        channel_name = next(
            (name for name, ch in self._channels.items() if ch is channel), "unknown"
        )

        loop = asyncio.get_event_loop()

        # Record delivery attempt
        def _record_attempt():
            self._db.execute(
                """
                INSERT INTO alert_delivery_status (
                    delivery_id, alert_id, channel, status, attempts
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (delivery_id, notification.alert_id, channel_name, DeliveryStatus.PENDING.value, 0),
            )
            self._db.commit()

        await loop.run_in_executor(None, _record_attempt)

        # Attempt delivery with retries
        for attempt in range(1, self.max_retry_attempts + 1):
            try:
                # Send notification
                if isinstance(channel, EmailNotifier) and address:
                    success = await channel.send(notification, address)
                else:
                    success = await channel.send(notification)

                # Update delivery status
                if success:
                    def _update_success():
                        self._db.execute(
                            """
                            UPDATE alert_delivery_status
                            SET status = ?, attempts = ?, delivered_at = ?
                            WHERE delivery_id = ?
                            """,
                            (DeliveryStatus.SUCCESS.value, attempt, time.time(), delivery_id),
                        )
                        self._db.commit()

                    await loop.run_in_executor(None, _update_success)
                    return True

                # Failed, update attempt count
                status = (
                    DeliveryStatus.RETRYING.value
                    if attempt < self.max_retry_attempts
                    else DeliveryStatus.FAILED.value
                )

                def _update_retry():
                    self._db.execute(
                        """
                        UPDATE alert_delivery_status
                        SET status = ?, attempts = ?, last_attempt_at = ?
                        WHERE delivery_id = ?
                        """,
                        (status, attempt, time.time(), delivery_id),
                    )
                    self._db.commit()

                await loop.run_in_executor(None, _update_retry)

                # Wait before retry (except on last attempt)
                if attempt < self.max_retry_attempts:
                    await asyncio.sleep(self.retry_delay_seconds)

            except Exception as e:
                logger.error(f"Error sending notification (attempt {attempt}): {e}")

                def _update_error():
                    self._db.execute(
                        """
                        UPDATE alert_delivery_status
                        SET status = ?, attempts = ?, last_attempt_at = ?, error_message = ?
                        WHERE delivery_id = ?
                        """,
                        (
                            DeliveryStatus.FAILED.value,
                            attempt,
                            time.time(),
                            str(e),
                            delivery_id,
                        ),
                    )
                    self._db.commit()

                await loop.run_in_executor(None, _update_error)

        return False

    def _row_to_notification(self, row: sqlite3.Row) -> AlertNotification:
        """Convert database row to AlertNotification."""
        return AlertNotification(
            alert_id=row["alert_id"],
            rule_name=row["rule_name"],
            severity=AlertSeverity(row["severity"]),
            message=row["message"],
            metric_name=row["metric_name"],
            metric_value=row["metric_value"],
            threshold_value=row["threshold_value"],
            threshold_operator=row["threshold_operator"],
            details=json.loads(row["details_json"]) if row["details_json"] else {},
            timestamp=datetime.fromtimestamp(row["timestamp"], tz=timezone.utc),
            acknowledged=bool(row["acknowledged"]),
            acknowledged_at=(
                datetime.fromtimestamp(row["acknowledged_at"], tz=timezone.utc)
                if row["acknowledged_at"]
                else None
            ),
            acknowledged_by=row["acknowledged_by"],
        )
