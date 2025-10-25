"""
Security Monitoring and Real-time Alert System for workspace-qdrant-mcp.

This module provides comprehensive security monitoring capabilities including:
- Real-time security event monitoring
- Performance metrics collection and analysis
- Alert generation and notification system
- Security dashboard data aggregation
- Threat intelligence integration
- Incident response coordination

Features:
- Multi-channel alerting (email, webhook, logs)
- Configurable alert thresholds and rules
- Security metrics aggregation and trends
- Real-time dashboard data
- Integration with threat detection system
- Automated incident escalation
"""

import asyncio
import hashlib
import json
import smtplib
import statistics
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart as MimeMultipart
from email.mime.text import MIMEText as MimeText
from enum import Enum
from pathlib import Path
from typing import Any

import aiohttp
from loguru import logger

from .threat_detection import SecurityEvent, ThreatDetection, ThreatLevel, ThreatType


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


class MetricType(Enum):
    """Types of security metrics."""

    THREAT_COUNT = "threat_count"
    AUTHENTICATION_FAILURES = "auth_failures"
    RATE_VIOLATIONS = "rate_violations"
    BLOCKED_IPS = "blocked_ips"
    QUERY_ANOMALIES = "query_anomalies"
    COLLECTION_ACCESS = "collection_access"
    ERROR_RATE = "error_rate"
    RESPONSE_TIME = "response_time"
    ACTIVE_SESSIONS = "active_sessions"


@dataclass
class SecurityAlert:
    """Represents a security alert."""

    alert_id: str
    alert_level: AlertLevel
    title: str
    description: str
    source: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    escalated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary format."""
        return {
            'alert_id': self.alert_id,
            'alert_level': self.alert_level.name,
            'title': self.title,
            'description': self.description,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'acknowledged': self.acknowledged,
            'escalated': self.escalated
        }


@dataclass
class SecurityMetric:
    """Represents a security metric data point."""

    metric_type: MetricType
    value: int | float
    timestamp: datetime
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metric to dictionary format."""
        return {
            'metric_type': self.metric_type.value,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags,
            'metadata': self.metadata
        }


class SecurityMetrics:
    """Collects and manages security metrics."""

    def __init__(self, retention_period: timedelta = timedelta(hours=24)):
        """
        Initialize security metrics collector.

        Args:
            retention_period: How long to retain metric data
        """
        self.retention_period = retention_period
        self.metrics: dict[MetricType, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.aggregated_metrics: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def record_metric(self, metric: SecurityMetric) -> None:
        """
        Record a security metric.

        Args:
            metric: Security metric to record
        """
        async with self._lock:
            self.metrics[metric.metric_type].append(metric)
            await self._cleanup_old_metrics()

    async def get_metric_summary(self,
                                metric_type: MetricType,
                                time_window: timedelta = timedelta(hours=1)) -> dict[str, Any]:
        """
        Get summary statistics for a metric type.

        Args:
            metric_type: Type of metric to summarize
            time_window: Time window for analysis

        Returns:
            Dictionary containing metric summary
        """
        async with self._lock:
            cutoff_time = datetime.utcnow() - time_window
            recent_metrics = [
                m for m in self.metrics[metric_type]
                if m.timestamp >= cutoff_time
            ]

            if not recent_metrics:
                return {
                    'count': 0,
                    'total': 0,
                    'average': 0,
                    'min': 0,
                    'max': 0,
                    'trend': 'stable'
                }

            values = [m.value for m in recent_metrics]
            summary = {
                'count': len(values),
                'total': sum(values),
                'average': statistics.mean(values),
                'min': min(values),
                'max': max(values),
                'std_dev': statistics.stdev(values) if len(values) > 1 else 0
            }

            # Calculate trend
            if len(values) >= 2:
                first_half = values[:len(values)//2]
                second_half = values[len(values)//2:]

                if statistics.mean(second_half) > statistics.mean(first_half) * 1.1:
                    summary['trend'] = 'increasing'
                elif statistics.mean(second_half) < statistics.mean(first_half) * 0.9:
                    summary['trend'] = 'decreasing'
                else:
                    summary['trend'] = 'stable'
            else:
                summary['trend'] = 'insufficient_data'

            return summary

    async def get_all_metric_summaries(self,
                                     time_window: timedelta = timedelta(hours=1)) -> dict[str, dict[str, Any]]:
        """Get summary for all metric types."""
        summaries = {}
        for metric_type in MetricType:
            summaries[metric_type.value] = await self.get_metric_summary(metric_type, time_window)
        return summaries

    async def _cleanup_old_metrics(self) -> None:
        """Remove metrics older than retention period."""
        cutoff_time = datetime.utcnow() - self.retention_period

        for _metric_type, metric_deque in self.metrics.items():
            # Remove old metrics from the left
            while metric_deque and metric_deque[0].timestamp < cutoff_time:
                metric_deque.popleft()


class AlertingSystem:
    """Manages alert generation, routing, and notification."""

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize alerting system.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.alerts: dict[str, SecurityAlert] = {}
        self.alert_rules: list[dict[str, Any]] = []
        self.notification_handlers: list[Callable] = []
        self.alert_history: deque = deque(maxlen=10000)

        # Alert suppression
        self.suppressed_alerts: dict[str, datetime] = {}
        self.suppression_duration = timedelta(minutes=self.config.get('suppression_minutes', 10))

        # Email configuration
        self.email_config = self.config.get('email', {})

        # Webhook configuration
        self.webhook_config = self.config.get('webhooks', {})

    def add_alert_rule(self, rule: dict[str, Any]) -> None:
        """
        Add an alert rule.

        Args:
            rule: Alert rule configuration
        """
        required_fields = ['name', 'condition', 'alert_level', 'message']
        if not all(field in rule for field in required_fields):
            raise ValueError(f"Alert rule must contain: {required_fields}")

        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule['name']}")

    def register_notification_handler(self, handler: Callable[[SecurityAlert], None]) -> None:
        """Register a custom notification handler."""
        self.notification_handlers.append(handler)

    async def evaluate_alert_rules(self, metrics: dict[str, dict[str, Any]],
                                 threats: list[ThreatDetection]) -> list[SecurityAlert]:
        """
        Evaluate alert rules against current metrics and threats.

        Args:
            metrics: Current metric summaries
            threats: Active threats

        Returns:
            List of generated alerts
        """
        generated_alerts = []

        # Evaluate threat-based alerts
        for threat in threats:
            alert = await self._create_threat_alert(threat)
            if alert and not self._is_suppressed(alert):
                generated_alerts.append(alert)

        # Evaluate metric-based alerts
        for rule in self.alert_rules:
            alert = await self._evaluate_metric_rule(rule, metrics)
            if alert and not self._is_suppressed(alert):
                generated_alerts.append(alert)

        return generated_alerts

    async def _create_threat_alert(self, threat: ThreatDetection) -> SecurityAlert | None:
        """Create alert from threat detection."""
        alert_level_mapping = {
            ThreatLevel.LOW: AlertLevel.WARNING,
            ThreatLevel.MEDIUM: AlertLevel.WARNING,
            ThreatLevel.HIGH: AlertLevel.ERROR,
            ThreatLevel.CRITICAL: AlertLevel.CRITICAL
        }

        # Generate unique alert ID based on threat
        alert_key = f"threat_{threat.threat_type.value}_{threat.threat_level.value}"
        alert_id = hashlib.md5(alert_key.encode()).hexdigest()[:12]

        return SecurityAlert(
            alert_id=alert_id,
            alert_level=alert_level_mapping[threat.threat_level],
            title=f"{threat.threat_type.value.replace('_', ' ').title()} Detected",
            description=threat.description,
            source="threat_detection",
            metadata={
                'threat_id': threat.threat_id,
                'threat_type': threat.threat_type.value,
                'confidence': threat.confidence,
                'mitigation_suggestions': threat.mitigation_suggestions
            }
        )

    async def _evaluate_metric_rule(self, rule: dict[str, Any],
                                  metrics: dict[str, dict[str, Any]]) -> SecurityAlert | None:
        """Evaluate a single metric-based alert rule."""
        try:
            condition = rule['condition']

            # Simple condition evaluation (can be extended)
            if self._evaluate_condition(condition, metrics):
                alert_id = hashlib.md5(f"{rule['name']}_{time.time()}".encode()).hexdigest()[:12]

                return SecurityAlert(
                    alert_id=alert_id,
                    alert_level=AlertLevel[rule['alert_level'].upper()],
                    title=rule['name'],
                    description=rule['message'],
                    source="metric_rule",
                    metadata={'rule': rule['name'], 'condition': condition}
                )

        except Exception as e:
            logger.error(f"Error evaluating alert rule {rule['name']}: {e}")

        return None

    def _evaluate_condition(self, condition: str, metrics: dict[str, dict[str, Any]]) -> bool:
        """Evaluate alert condition against metrics."""
        try:
            # Simple condition parser - can be extended for complex expressions
            # Example: "threat_count.total > 10"
            parts = condition.split()
            if len(parts) >= 3:
                metric_path = parts[0]  # e.g., "threat_count.total"
                operator = parts[1]     # e.g., ">"
                threshold = float(parts[2])  # e.g., "10"

                # Navigate to metric value
                path_parts = metric_path.split('.')
                value = metrics

                for part in path_parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        return False

                # Evaluate condition
                if operator == '>':
                    return float(value) > threshold
                elif operator == '<':
                    return float(value) < threshold
                elif operator == '>=':
                    return float(value) >= threshold
                elif operator == '<=':
                    return float(value) <= threshold
                elif operator == '==':
                    return float(value) == threshold
                elif operator == '!=':
                    return float(value) != threshold

        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"Error evaluating condition '{condition}': {e}")

        return False

    def _is_suppressed(self, alert: SecurityAlert) -> bool:
        """Check if alert should be suppressed."""
        suppression_key = f"{alert.source}_{alert.title}"

        if suppression_key in self.suppressed_alerts:
            last_sent = self.suppressed_alerts[suppression_key]
            if datetime.utcnow() - last_sent < self.suppression_duration:
                return True

        return False

    async def send_alert(self, alert: SecurityAlert) -> None:
        """
        Send alert through configured notification channels.

        Args:
            alert: Security alert to send
        """
        # Store alert
        self.alerts[alert.alert_id] = alert
        self.alert_history.append(alert)

        # Mark as sent to prevent spam
        suppression_key = f"{alert.source}_{alert.title}"
        self.suppressed_alerts[suppression_key] = datetime.utcnow()

        try:
            # Send through configured channels
            await self._send_email_notification(alert)
            await self._send_webhook_notification(alert)

            # Call custom handlers
            for handler in self.notification_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(alert)
                    else:
                        handler(alert)
                except Exception as e:
                    logger.error(f"Error in notification handler: {e}")

            logger.info(f"Alert sent: {alert.title} (Level: {alert.alert_level.name})")

        except Exception as e:
            logger.error(f"Error sending alert {alert.alert_id}: {e}")

    async def _send_email_notification(self, alert: SecurityAlert) -> None:
        """Send alert via email."""
        if not self.email_config.get('enabled', False):
            return

        try:
            smtp_server = self.email_config['smtp_server']
            smtp_port = self.email_config.get('smtp_port', 587)
            username = self.email_config['username']
            password = self.email_config['password']
            recipients = self.email_config['recipients']

            # Only send high priority alerts via email unless configured otherwise
            min_level = AlertLevel[self.email_config.get('min_level', 'ERROR').upper()]
            if alert.alert_level.value < min_level.value:
                return

            msg = MimeMultipart()
            msg['From'] = username
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"[SECURITY ALERT] {alert.title}"

            body = f"""
Security Alert Generated

Alert ID: {alert.alert_id}
Level: {alert.alert_level.name}
Source: {alert.source}
Timestamp: {alert.timestamp}

Description:
{alert.description}

Metadata:
{json.dumps(alert.metadata, indent=2)}

Please review and take appropriate action.
"""

            msg.attach(MimeText(body, 'plain'))

            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(msg)

            logger.info(f"Email alert sent for {alert.alert_id}")

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    async def _send_webhook_notification(self, alert: SecurityAlert) -> None:
        """Send alert via webhook."""
        webhooks = self.webhook_config.get('urls', [])
        if not webhooks:
            return

        payload = alert.to_dict()

        async with aiohttp.ClientSession() as session:
            for webhook_url in webhooks:
                try:
                    async with session.post(
                        webhook_url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        if response.status == 200:
                            logger.info(f"Webhook alert sent to {webhook_url}")
                        else:
                            logger.warning(f"Webhook returned status {response.status}")

                except Exception as e:
                    logger.error(f"Failed to send webhook to {webhook_url}: {e}")

    def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: ID of alert to acknowledge
            user: User acknowledging the alert

        Returns:
            True if alert was acknowledged
        """
        if alert_id in self.alerts:
            self.alerts[alert_id].acknowledged = True
            self.alerts[alert_id].metadata['acknowledged_by'] = user
            self.alerts[alert_id].metadata['acknowledged_at'] = datetime.utcnow().isoformat()
            logger.info(f"Alert {alert_id} acknowledged by {user}")
            return True

        return False

    def get_active_alerts(self, alert_level: AlertLevel | None = None) -> list[SecurityAlert]:
        """Get currently active (unacknowledged) alerts."""
        active = [alert for alert in self.alerts.values() if not alert.acknowledged]

        if alert_level:
            active = [alert for alert in active if alert.alert_level == alert_level]

        return sorted(active, key=lambda a: a.timestamp, reverse=True)


class SecurityEventLogger:
    """Centralized security event logging."""

    def __init__(self, log_file: Path | None = None):
        """
        Initialize security event logger.

        Args:
            log_file: Optional custom log file path
        """
        self.log_file = log_file or Path("security_events.log")
        self.event_buffer: deque = deque(maxlen=10000)
        self._lock = asyncio.Lock()

    async def log_event(self, event: SecurityEvent) -> None:
        """
        Log a security event.

        Args:
            event: Security event to log
        """
        async with self._lock:
            log_entry = {
                'timestamp': event.timestamp.isoformat(),
                'event_id': event.event_id,
                'event_type': event.event_type,
                'source_ip': event.source_ip,
                'user_id': event.user_id,
                'collection': event.collection,
                'query_hash': hashlib.md5(event.query.encode()).hexdigest() if event.query else None,
                'metadata': event.metadata
            }

            self.event_buffer.append(log_entry)

            # Write to file (in production, use proper logging framework)
            try:
                with open(self.log_file, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')
            except Exception as e:
                logger.error(f"Failed to write to security log: {e}")

    async def log_threat(self, threat: ThreatDetection) -> None:
        """
        Log a threat detection.

        Args:
            threat: Threat detection to log
        """
        async with self._lock:
            log_entry = {
                'timestamp': threat.timestamp.isoformat(),
                'threat_id': threat.threat_id,
                'threat_type': threat.threat_type.value,
                'threat_level': threat.threat_level.value,
                'confidence': threat.confidence,
                'description': threat.description,
                'source_event_count': len(threat.source_events),
                'mitigation_suggestions': threat.mitigation_suggestions
            }

            self.event_buffer.append(log_entry)

            try:
                with open(self.log_file, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')
            except Exception as e:
                logger.error(f"Failed to write threat log: {e}")

    def get_recent_events(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent security events from buffer."""
        return list(self.event_buffer)[-limit:]


class SecurityMonitor:
    """Main security monitoring coordinator."""

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize security monitor.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Initialize components
        self.metrics = SecurityMetrics(
            retention_period=timedelta(hours=self.config.get('metric_retention_hours', 24))
        )

        self.alerting = AlertingSystem(self.config.get('alerting', {}))

        self.event_logger = SecurityEventLogger(
            log_file=Path(self.config.get('log_file', 'security_events.log'))
        )

        # Monitoring state
        self.active_threats: list[ThreatDetection] = []
        self.monitoring_active = False
        self.monitor_task: asyncio.Task | None = None

        # Performance tracking
        self.last_check_time = datetime.utcnow()
        self.check_interval = timedelta(seconds=self.config.get('check_interval_seconds', 30))

        self._setup_default_alert_rules()

    def _setup_default_alert_rules(self) -> None:
        """Setup default alert rules."""
        default_rules = [
            {
                'name': 'High Threat Count',
                'condition': 'threat_count.total > 5',
                'alert_level': 'ERROR',
                'message': 'High number of security threats detected in the last hour'
            },
            {
                'name': 'Authentication Failures',
                'condition': 'auth_failures.total > 10',
                'alert_level': 'WARNING',
                'message': 'High number of authentication failures detected'
            },
            {
                'name': 'Critical Threat Detected',
                'condition': 'threat_count.max >= 4',  # ThreatLevel.CRITICAL = 4
                'alert_level': 'CRITICAL',
                'message': 'Critical security threat detected - immediate attention required'
            }
        ]

        for rule in default_rules:
            self.alerting.add_alert_rule(rule)

    async def record_security_event(self, event: SecurityEvent) -> None:
        """
        Record a security event for monitoring.

        Args:
            event: Security event to record
        """
        # Log the event
        await self.event_logger.log_event(event)

        # Update metrics
        await self.metrics.record_metric(SecurityMetric(
            metric_type=MetricType.COLLECTION_ACCESS,
            value=1,
            timestamp=event.timestamp,
            tags={'collection': event.collection or 'unknown', 'user': event.user_id or 'anonymous'}
        ))

    async def record_threat(self, threat: ThreatDetection) -> None:
        """
        Record a threat detection.

        Args:
            threat: Threat detection to record
        """
        # Add to active threats
        self.active_threats.append(threat)

        # Log the threat
        await self.event_logger.log_threat(threat)

        # Update threat metrics
        await self.metrics.record_metric(SecurityMetric(
            metric_type=MetricType.THREAT_COUNT,
            value=threat.threat_level.value,
            timestamp=threat.timestamp,
            tags={'threat_type': threat.threat_type.value, 'threat_level': threat.threat_level.name}
        ))

    async def start_monitoring(self) -> None:
        """Start the security monitoring loop."""
        if self.monitoring_active:
            logger.warning("Security monitoring already active")
            return

        self.monitoring_active = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Security monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop the security monitoring loop."""
        self.monitoring_active = False

        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("Security monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                await self._perform_monitoring_check()
                await asyncio.sleep(self.check_interval.total_seconds())

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Short delay before retrying

    async def _perform_monitoring_check(self) -> None:
        """Perform a monitoring check cycle."""
        current_time = datetime.utcnow()

        # Get current metrics
        metric_summaries = await self.metrics.get_all_metric_summaries(
            time_window=timedelta(hours=1)
        )

        # Clean up old threats (older than 1 hour)
        cutoff_time = current_time - timedelta(hours=1)
        self.active_threats = [
            t for t in self.active_threats
            if t.timestamp >= cutoff_time
        ]

        # Evaluate alert rules
        alerts = await self.alerting.evaluate_alert_rules(
            metrics=metric_summaries,
            threats=self.active_threats
        )

        # Send generated alerts
        for alert in alerts:
            await self.alerting.send_alert(alert)

        self.last_check_time = current_time

    async def get_security_dashboard_data(self) -> dict[str, Any]:
        """Get comprehensive security dashboard data."""
        current_time = datetime.utcnow()

        # Get metric summaries for different time windows
        hourly_metrics = await self.metrics.get_all_metric_summaries(timedelta(hours=1))
        daily_metrics = await self.metrics.get_all_metric_summaries(timedelta(hours=24))

        # Active threat summary
        threat_summary = {
            'total_active': len(self.active_threats),
            'by_level': {
                level.name: len([t for t in self.active_threats if t.threat_level == level])
                for level in ThreatLevel
            },
            'by_type': {
                threat_type.name: len([t for t in self.active_threats if t.threat_type == threat_type])
                for threat_type in ThreatType
            }
        }

        # Alert summary
        active_alerts = self.alerting.get_active_alerts()
        alert_summary = {
            'total_active': len(active_alerts),
            'by_level': {
                level.name: len([a for a in active_alerts if a.alert_level == level])
                for level in AlertLevel
            }
        }

        # Recent events
        recent_events = self.event_logger.get_recent_events(limit=50)

        return {
            'timestamp': current_time.isoformat(),
            'monitoring_status': {
                'active': self.monitoring_active,
                'last_check': self.last_check_time.isoformat(),
                'check_interval_seconds': self.check_interval.total_seconds()
            },
            'metrics': {
                'hourly': hourly_metrics,
                'daily': daily_metrics
            },
            'threats': threat_summary,
            'alerts': alert_summary,
            'recent_events': recent_events[-10:]  # Last 10 events for dashboard
        }

    def get_system_health(self) -> dict[str, Any]:
        """Get security monitoring system health status."""
        return {
            'monitoring_active': self.monitoring_active,
            'last_check': self.last_check_time.isoformat(),
            'active_threats': len(self.active_threats),
            'active_alerts': len(self.alerting.get_active_alerts()),
            'metric_types': len(MetricType),
            'alert_rules': len(self.alerting.alert_rules),
            'event_buffer_size': len(self.event_logger.event_buffer)
        }


# Export public interface
__all__ = [
    'SecurityMonitor',
    'SecurityMetrics',
    'AlertingSystem',
    'SecurityEventLogger',
    'SecurityAlert',
    'SecurityMetric',
    'AlertLevel',
    'MetricType'
]
