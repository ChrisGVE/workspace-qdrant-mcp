"""Alerting System - Comprehensive anomaly alerting and notification framework."""

import time
import logging
from typing import Dict, List, Optional, Any, Callable, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import threading
from queue import Queue, Empty
import json
import smtplib
import requests
from collections import defaultdict, deque

from .detection_algorithms import AnomalyResult


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts."""
    THRESHOLD = "threshold"
    PATTERN = "pattern"
    TREND = "trend"
    STATISTICAL = "statistical"
    SYSTEM = "system"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class AlertRule:
    """Alert rule configuration."""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    alert_type: AlertType
    conditions: Dict[str, Any]
    enabled: bool = True
    cooldown_minutes: int = 30
    escalation_minutes: Optional[int] = None
    tags: Set[str] = field(default_factory=set)

    def __post_init__(self):
        """Validate rule configuration."""
        if not self.id:
            raise ValueError("Alert rule ID cannot be empty")
        if not self.name:
            raise ValueError("Alert rule name cannot be empty")
        if self.cooldown_minutes < 0:
            raise ValueError("Cooldown minutes must be non-negative")
        if self.escalation_minutes is not None and self.escalation_minutes < 0:
            raise ValueError("Escalation minutes must be non-negative")


@dataclass
class Alert:
    """Individual alert instance."""
    id: str
    rule_id: str
    title: str
    description: str
    severity: AlertSeverity
    alert_type: AlertType
    status: AlertStatus
    timestamp: datetime
    anomaly_result: Optional[AnomalyResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    escalated: bool = False
    escalated_at: Optional[datetime] = None
    suppressed_until: Optional[datetime] = None

    def __post_init__(self):
        """Validate alert data."""
        if not self.id:
            raise ValueError("Alert ID cannot be empty")
        if not self.rule_id:
            raise ValueError("Alert rule ID cannot be empty")
        if not self.title:
            raise ValueError("Alert title cannot be empty")


class NotificationChannel:
    """Base class for notification channels."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.enabled = config.get('enabled', True)
        self.rate_limit = config.get('rate_limit_per_hour', 100)
        self.sent_count = 0
        self.last_reset = time.time()

    def can_send(self) -> bool:
        """Check if notification can be sent (rate limiting)."""
        current_time = time.time()
        if current_time - self.last_reset >= 3600:  # Reset hourly
            self.sent_count = 0
            self.last_reset = current_time

        return self.enabled and self.sent_count < self.rate_limit

    def send(self, alert: Alert) -> bool:
        """Send notification for alert."""
        if not self.can_send():
            return False

        try:
            result = self._send_notification(alert)
            if result:
                self.sent_count += 1
            return result
        except Exception as e:
            logging.error(f"Failed to send notification via {self.name}: {e}")
            return False

    def _send_notification(self, alert: Alert) -> bool:
        """Implement specific notification sending logic."""
        raise NotImplementedError("Subclasses must implement _send_notification")


class EmailNotificationChannel(NotificationChannel):
    """Email notification channel."""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        required_keys = ['smtp_server', 'smtp_port', 'username', 'password', 'recipients']
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Email configuration missing keys: {missing_keys}")

    def _send_notification(self, alert: Alert) -> bool:
        """Send email notification."""
        try:
            # Create simple email message
            from_addr = self.config['username']
            to_addrs = self.config['recipients']
            subject = f"[{alert.severity.value.upper()}] {alert.title}"
            body = self._format_email_body(alert)

            # Create message string
            message = f"""From: {from_addr}
To: {', '.join(to_addrs)}
Subject: {subject}
Content-Type: text/html

{body}"""

            with smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port']) as server:
                server.starttls()
                server.login(self.config['username'], self.config['password'])
                server.sendmail(from_addr, to_addrs, message)

            return True
        except Exception as e:
            logging.error(f"Email send failed: {e}")
            return False

    def _format_email_body(self, alert: Alert) -> str:
        """Format email body."""
        anomaly_info = ""
        if alert.anomaly_result:
            anomaly_info = f"""
            <h3>Anomaly Details</h3>
            <ul>
                <li><strong>Score:</strong> {alert.anomaly_result.anomaly_score:.4f}</li>
                <li><strong>Confidence:</strong> {alert.anomaly_result.confidence:.4f}</li>
                <li><strong>Method:</strong> {alert.anomaly_result.method}</li>
                <li><strong>Expected:</strong> {alert.anomaly_result.expected_value}</li>
                <li><strong>Actual:</strong> {alert.anomaly_result.actual_value}</li>
            </ul>
            """

        return f"""
        <html>
        <body>
            <h2>Alert: {alert.title}</h2>
            <p><strong>Severity:</strong> {alert.severity.value.upper()}</p>
            <p><strong>Type:</strong> {alert.alert_type.value}</p>
            <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Description:</strong> {alert.description}</p>
            {anomaly_info}
            <hr>
            <p><em>Alert ID: {alert.id}</em></p>
        </body>
        </html>
        """


class WebhookNotificationChannel(NotificationChannel):
    """Webhook notification channel."""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        if 'url' not in config:
            raise ValueError("Webhook configuration missing 'url'")

    def _send_notification(self, alert: Alert) -> bool:
        """Send webhook notification."""
        try:
            payload = {
                'alert_id': alert.id,
                'rule_id': alert.rule_id,
                'title': alert.title,
                'description': alert.description,
                'severity': alert.severity.value,
                'alert_type': alert.alert_type.value,
                'status': alert.status.value,
                'timestamp': alert.timestamp.isoformat(),
                'metadata': alert.metadata
            }

            if alert.anomaly_result:
                payload['anomaly'] = {
                    'score': alert.anomaly_result.anomaly_score,
                    'confidence': alert.anomaly_result.confidence,
                    'method': alert.anomaly_result.method,
                    'expected_value': alert.anomaly_result.expected_value,
                    'actual_value': alert.anomaly_result.actual_value,
                    'threshold': alert.anomaly_result.threshold
                }

            headers = {'Content-Type': 'application/json'}
            if 'headers' in self.config:
                headers.update(self.config['headers'])

            response = requests.post(
                self.config['url'],
                json=payload,
                headers=headers,
                timeout=self.config.get('timeout', 30)
            )

            return response.status_code < 400
        except Exception as e:
            logging.error(f"Webhook send failed: {e}")
            return False


class SlackNotificationChannel(NotificationChannel):
    """Slack notification channel."""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        if 'webhook_url' not in config:
            raise ValueError("Slack configuration missing 'webhook_url'")

    def _send_notification(self, alert: Alert) -> bool:
        """Send Slack notification."""
        try:
            color_map = {
                AlertSeverity.LOW: "#36a64f",
                AlertSeverity.MEDIUM: "#ff9900",
                AlertSeverity.HIGH: "#ff6600",
                AlertSeverity.CRITICAL: "#ff0000"
            }

            attachment = {
                "color": color_map.get(alert.severity, "#808080"),
                "title": alert.title,
                "text": alert.description,
                "fields": [
                    {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                    {"title": "Type", "value": alert.alert_type.value, "short": True},
                    {"title": "Time", "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), "short": True},
                    {"title": "Alert ID", "value": alert.id, "short": True}
                ],
                "footer": "Anomaly Detection System",
                "ts": int(alert.timestamp.timestamp())
            }

            if alert.anomaly_result:
                attachment["fields"].extend([
                    {"title": "Anomaly Score", "value": f"{alert.anomaly_result.anomaly_score:.4f}", "short": True},
                    {"title": "Confidence", "value": f"{alert.anomaly_result.confidence:.4f}", "short": True}
                ])

            payload = {
                "attachments": [attachment],
                "channel": self.config.get('channel', '#alerts')
            }

            response = requests.post(
                self.config['webhook_url'],
                json=payload,
                timeout=30
            )

            return response.status_code == 200
        except Exception as e:
            logging.error(f"Slack send failed: {e}")
            return False


class AlertingSystem:
    """Comprehensive alerting system for anomaly detection."""

    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.notification_channels: Dict[str, NotificationChannel] = {}
        self.rule_channels: Dict[str, List[str]] = {}
        self.suppressed_rules: Dict[str, datetime] = {}
        self.cooldown_tracker: Dict[str, datetime] = {}

        # Alert processing queue
        self.alert_queue: Queue = Queue()
        self.processing_thread: Optional[threading.Thread] = None
        self.running = False

        # Statistics
        self.stats = {
            'total_alerts': 0,
            'active_alerts': 0,
            'alerts_by_severity': defaultdict(int),
            'alerts_by_type': defaultdict(int),
            'notifications_sent': 0,
            'notifications_failed': 0
        }

        self.logger = logging.getLogger(__name__)

    def add_rule(
        self,
        rule_id: str,
        name: str,
        description: str,
        severity: AlertSeverity,
        alert_type: AlertType,
        conditions: Dict[str, Any],
        enabled: bool = True,
        cooldown_minutes: int = 30,
        escalation_minutes: Optional[int] = None,
        tags: Optional[Set[str]] = None
    ) -> AlertRule:
        """Add alert rule."""
        if rule_id in self.rules:
            raise ValueError(f"Rule with ID {rule_id} already exists")

        rule = AlertRule(
            id=rule_id,
            name=name,
            description=description,
            severity=severity,
            alert_type=alert_type,
            conditions=conditions,
            enabled=enabled,
            cooldown_minutes=cooldown_minutes,
            escalation_minutes=escalation_minutes,
            tags=tags or set()
        )

        self.rules[rule_id] = rule
        self.logger.info(f"Added alert rule: {rule_id}")
        return rule

    def remove_rule(self, rule_id: str) -> bool:
        """Remove alert rule."""
        if rule_id not in self.rules:
            return False

        # Resolve any active alerts for this rule
        alerts_to_resolve = [
            alert for alert in self.active_alerts.values()
            if alert.rule_id == rule_id and alert.status == AlertStatus.ACTIVE
        ]

        for alert in alerts_to_resolve:
            self.resolve_alert(alert.id, "Rule removed")

        del self.rules[rule_id]
        self.rule_channels.pop(rule_id, None)
        self.cooldown_tracker.pop(rule_id, None)

        self.logger.info(f"Removed alert rule: {rule_id}")
        return True

    def enable_rule(self, rule_id: str) -> bool:
        """Enable alert rule."""
        if rule_id not in self.rules:
            return False

        self.rules[rule_id].enabled = True
        self.logger.info(f"Enabled alert rule: {rule_id}")
        return True

    def disable_rule(self, rule_id: str) -> bool:
        """Disable alert rule."""
        if rule_id not in self.rules:
            return False

        self.rules[rule_id].enabled = False

        # Resolve active alerts for this rule
        alerts_to_resolve = [
            alert for alert in self.active_alerts.values()
            if alert.rule_id == rule_id and alert.status == AlertStatus.ACTIVE
        ]

        for alert in alerts_to_resolve:
            self.resolve_alert(alert.id, "Rule disabled")

        self.logger.info(f"Disabled alert rule: {rule_id}")
        return True

    def add_notification_channel(
        self,
        channel_name: str,
        channel_type: str,
        config: Dict[str, Any]
    ) -> bool:
        """Add notification channel."""
        try:
            if channel_type == 'email':
                channel = EmailNotificationChannel(channel_name, config)
            elif channel_type == 'webhook':
                channel = WebhookNotificationChannel(channel_name, config)
            elif channel_type == 'slack':
                channel = SlackNotificationChannel(channel_name, config)
            else:
                raise ValueError(f"Unsupported channel type: {channel_type}")

            self.notification_channels[channel_name] = channel
            self.logger.info(f"Added notification channel: {channel_name} ({channel_type})")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add notification channel {channel_name}: {e}")
            return False

    def assign_channels_to_rule(self, rule_id: str, channel_names: List[str]) -> bool:
        """Assign notification channels to rule."""
        if rule_id not in self.rules:
            return False

        # Validate channels exist
        missing_channels = [name for name in channel_names if name not in self.notification_channels]
        if missing_channels:
            raise ValueError(f"Unknown notification channels: {missing_channels}")

        self.rule_channels[rule_id] = channel_names
        self.logger.info(f"Assigned channels {channel_names} to rule {rule_id}")
        return True

    def evaluate_anomaly(self, anomaly_result: AnomalyResult, context: Dict[str, Any] = None) -> List[Alert]:
        """Evaluate anomaly result against alert rules."""
        if context is None:
            context = {}

        triggered_alerts = []

        for rule in self.rules.values():
            if not rule.enabled:
                continue

            # Check if rule is in cooldown
            if self._is_rule_in_cooldown(rule.id):
                continue

            # Check if rule matches anomaly
            if self._evaluate_rule_conditions(rule, anomaly_result, context):
                alert = self._create_alert(rule, anomaly_result, context)
                triggered_alerts.append(alert)
                self._queue_alert_for_processing(alert)

        return triggered_alerts

    def _evaluate_rule_conditions(
        self,
        rule: AlertRule,
        anomaly_result: AnomalyResult,
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate if rule conditions are met."""
        conditions = rule.conditions

        # Check anomaly score threshold
        if 'min_anomaly_score' in conditions:
            if anomaly_result.anomaly_score < conditions['min_anomaly_score']:
                return False

        # Check confidence threshold
        if 'min_confidence' in conditions:
            if anomaly_result.confidence < conditions['min_confidence']:
                return False

        # Check detection method
        if 'methods' in conditions:
            if anomaly_result.method not in conditions['methods']:
                return False

        # Check metric pattern
        if 'metric_patterns' in conditions:
            metric_name = context.get('metric_name', '')
            if not any(pattern in metric_name for pattern in conditions['metric_patterns']):
                return False

        # Check value deviation
        if 'min_deviation_percent' in conditions and anomaly_result.expected_value is not None:
            expected = anomaly_result.expected_value
            actual = anomaly_result.actual_value
            if expected != 0:
                deviation_percent = abs((actual - expected) / expected) * 100
                if deviation_percent < conditions['min_deviation_percent']:
                    return False

        # Check context tags
        if 'required_tags' in conditions:
            context_tags = set(context.get('tags', []))
            required_tags = set(conditions['required_tags'])
            if not required_tags.issubset(context_tags):
                return False

        return True

    def _create_alert(
        self,
        rule: AlertRule,
        anomaly_result: AnomalyResult,
        context: Dict[str, Any]
    ) -> Alert:
        """Create alert from rule and anomaly result."""
        alert_id = f"{rule.id}_{int(time.time() * 1000)}"

        # Format title and description
        title = rule.name
        if anomaly_result.expected_value is not None:
            title += f" (Expected: {anomaly_result.expected_value:.2f}, Actual: {anomaly_result.actual_value:.2f})"

        description = rule.description
        if 'metric_name' in context:
            description += f" Metric: {context['metric_name']}"

        alert = Alert(
            id=alert_id,
            rule_id=rule.id,
            title=title,
            description=description,
            severity=rule.severity,
            alert_type=rule.alert_type,
            status=AlertStatus.ACTIVE,
            timestamp=datetime.now(),
            anomaly_result=anomaly_result,
            metadata=context.copy()
        )

        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)

        # Update statistics
        self.stats['total_alerts'] += 1
        self.stats['active_alerts'] += 1
        self.stats['alerts_by_severity'][rule.severity.value] += 1
        self.stats['alerts_by_type'][rule.alert_type.value] += 1

        # Set cooldown
        self.cooldown_tracker[rule.id] = datetime.now() + timedelta(minutes=rule.cooldown_minutes)

        self.logger.info(f"Created alert: {alert_id} for rule {rule.id}")
        return alert

    def _queue_alert_for_processing(self, alert: Alert):
        """Queue alert for notification processing."""
        self.alert_queue.put(alert)

    def _is_rule_in_cooldown(self, rule_id: str) -> bool:
        """Check if rule is in cooldown period."""
        if rule_id not in self.cooldown_tracker:
            return False

        return datetime.now() < self.cooldown_tracker[rule_id]

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = None) -> bool:
        """Acknowledge alert."""
        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        if alert.status != AlertStatus.ACTIVE:
            return False

        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_at = datetime.now()

        self.logger.info(f"Acknowledged alert: {alert_id}")
        return True

    def resolve_alert(self, alert_id: str, resolution_note: str = None) -> bool:
        """Resolve alert."""
        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now()

        if resolution_note:
            alert.metadata['resolution_note'] = resolution_note

        self.stats['active_alerts'] -= 1
        self.logger.info(f"Resolved alert: {alert_id}")
        return True

    def suppress_rule(self, rule_id: str, duration_minutes: int) -> bool:
        """Suppress rule for specified duration."""
        if rule_id not in self.rules:
            return False

        until_time = datetime.now() + timedelta(minutes=duration_minutes)
        self.suppressed_rules[rule_id] = until_time

        # Suppress active alerts for this rule
        for alert in self.active_alerts.values():
            if alert.rule_id == rule_id and alert.status == AlertStatus.ACTIVE:
                alert.status = AlertStatus.SUPPRESSED
                alert.suppressed_until = until_time

        self.logger.info(f"Suppressed rule {rule_id} for {duration_minutes} minutes")
        return True

    def start_processing(self):
        """Start background alert processing."""
        if self.running:
            return

        self.running = True
        self.processing_thread = threading.Thread(target=self._process_alerts)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        self.logger.info("Started alert processing")

    def stop_processing(self):
        """Stop background alert processing."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        self.logger.info("Stopped alert processing")

    def _process_alerts(self):
        """Background alert processing loop."""
        while self.running:
            try:
                alert = self.alert_queue.get(timeout=1)
                self._send_notifications(alert)
                self._check_escalation(alert)
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing alert: {e}")

    def _send_notifications(self, alert: Alert):
        """Send notifications for alert."""
        channels = self.rule_channels.get(alert.rule_id, [])
        if not channels:
            return

        for channel_name in channels:
            if channel_name not in self.notification_channels:
                continue

            channel = self.notification_channels[channel_name]
            try:
                if channel.send(alert):
                    self.stats['notifications_sent'] += 1
                    self.logger.info(f"Sent notification for alert {alert.id} via {channel_name}")
                else:
                    self.stats['notifications_failed'] += 1
                    self.logger.warning(f"Failed to send notification for alert {alert.id} via {channel_name}")
            except Exception as e:
                self.stats['notifications_failed'] += 1
                self.logger.error(f"Error sending notification via {channel_name}: {e}")

    def _check_escalation(self, alert: Alert):
        """Check if alert should be escalated."""
        rule = self.rules.get(alert.rule_id)
        if not rule or not rule.escalation_minutes or alert.escalated:
            return

        if alert.status != AlertStatus.ACTIVE:
            return

        escalation_time = alert.timestamp + timedelta(minutes=rule.escalation_minutes)
        if datetime.now() >= escalation_time:
            alert.escalated = True
            alert.escalated_at = datetime.now()

            # Create escalated alert with higher severity
            escalated_severity = AlertSeverity.CRITICAL
            escalated_alert = Alert(
                id=f"{alert.id}_escalated",
                rule_id=alert.rule_id,
                title=f"ESCALATED: {alert.title}",
                description=f"Alert not acknowledged within {rule.escalation_minutes} minutes. {alert.description}",
                severity=escalated_severity,
                alert_type=alert.alert_type,
                status=AlertStatus.ACTIVE,
                timestamp=datetime.now(),
                anomaly_result=alert.anomaly_result,
                metadata=alert.metadata.copy()
            )

            self.active_alerts[escalated_alert.id] = escalated_alert
            self.alert_history.append(escalated_alert)
            self.stats['total_alerts'] += 1
            self.stats['active_alerts'] += 1
            self.stats['alerts_by_severity'][escalated_severity.value] += 1

            self._queue_alert_for_processing(escalated_alert)
            self.logger.warning(f"Escalated alert: {alert.id}")

    def get_active_alerts(
        self,
        severity_filter: Optional[AlertSeverity] = None,
        rule_filter: Optional[str] = None
    ) -> List[Alert]:
        """Get active alerts with optional filtering."""
        alerts = [alert for alert in self.active_alerts.values() if alert.status == AlertStatus.ACTIVE]

        if severity_filter:
            alerts = [alert for alert in alerts if alert.severity == severity_filter]

        if rule_filter:
            alerts = [alert for alert in alerts if rule_filter in alert.rule_id]

        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)

    def get_alert_history(
        self,
        limit: int = 100,
        severity_filter: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        """Get alert history with optional filtering."""
        alerts = list(self.alert_history)

        if severity_filter:
            alerts = [alert for alert in alerts if alert.severity == severity_filter]

        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)[:limit]

    def get_statistics(self) -> Dict[str, Any]:
        """Get alerting system statistics."""
        # Clean up old suppressed rules
        current_time = datetime.now()
        expired_suppressions = [
            rule_id for rule_id, until_time in self.suppressed_rules.items()
            if current_time > until_time
        ]
        for rule_id in expired_suppressions:
            del self.suppressed_rules[rule_id]

        stats = self.stats.copy()
        stats['rules_count'] = len(self.rules)
        stats['enabled_rules'] = len([r for r in self.rules.values() if r.enabled])
        stats['suppressed_rules'] = len(self.suppressed_rules)
        stats['notification_channels'] = len(self.notification_channels)
        stats['queue_size'] = self.alert_queue.qsize()

        return stats

    def cleanup_resolved_alerts(self, days_old: int = 30):
        """Clean up resolved alerts older than specified days."""
        cutoff_time = datetime.now() - timedelta(days=days_old)
        alerts_to_remove = []

        for alert_id, alert in self.active_alerts.items():
            if (alert.status == AlertStatus.RESOLVED and
                alert.resolved_at and
                alert.resolved_at < cutoff_time):
                alerts_to_remove.append(alert_id)

        for alert_id in alerts_to_remove:
            del self.active_alerts[alert_id]

        self.logger.info(f"Cleaned up {len(alerts_to_remove)} resolved alerts")
        return len(alerts_to_remove)