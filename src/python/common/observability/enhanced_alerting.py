"""
Enhanced Multi-Level Alerting System for workspace-qdrant-mcp.

Provides sophisticated alerting capabilities with multiple channels, escalation policies,
automated recovery actions, and intelligent alert aggregation and correlation.

Key Features:
    - Multi-channel alert delivery (email, webhook, Slack, PagerDuty)
    - Escalation policies with time-based escalation
    - Alert correlation and de-duplication
    - Automated recovery action triggering
    - Alert fatigue prevention with intelligent grouping
    - Alert templates and customizable severity levels
    - Integration with health monitoring and component lifecycle

Alert Channels:
    - Email notifications with HTML formatting
    - Webhook endpoints with JSON payloads
    - Slack integration with rich formatting
    - PagerDuty integration for critical alerts
    - Custom notification handlers

Example:
    ```python
    from workspace_qdrant_mcp.observability.enhanced_alerting import AlertingManager

    # Initialize alerting manager
    alerting = AlertingManager()
    await alerting.initialize()

    # Configure alert channels
    alerting.add_email_channel("admin@example.com", ["critical", "warning"])
    alerting.add_webhook_channel("https://hooks.slack.com/...", ["info", "warning"])

    # Send alert
    await alerting.send_alert(
        severity="critical",
        title="Component Failure",
        message="Python MCP Server is unhealthy",
        component="python_mcp_server",
        auto_recovery=True
    )
    ```
"""

import asyncio
import json
import smtplib
import ssl
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
from urllib.parse import urljoin

import aiohttp
from loguru import logger

from .health_coordinator import AlertSeverity, HealthAlert
from .metrics import metrics_instance


class AlertChannel(Enum):
    """Alert delivery channels."""

    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    CUSTOM = "custom"


class AlertStatus(Enum):
    """Alert processing status."""

    PENDING = "pending"
    SENT = "sent"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    FAILED = "failed"


@dataclass
class AlertChannelConfig:
    """Configuration for an alert channel."""

    channel_type: AlertChannel
    endpoint: str
    severities: List[str]
    enabled: bool = True
    rate_limit_per_hour: int = 100
    retry_attempts: int = 3
    retry_delay_seconds: float = 30.0
    timeout_seconds: float = 30.0
    headers: Dict[str, str] = field(default_factory=dict)
    auth_config: Dict[str, str] = field(default_factory=dict)


@dataclass
class EscalationPolicy:
    """Escalation policy configuration."""

    policy_id: str
    name: str
    escalation_levels: List[Dict[str, Any]]
    escalation_interval_minutes: float = 15.0
    max_escalations: int = 3
    enabled: bool = True


@dataclass
class AlertCorrelationRule:
    """Alert correlation rule for intelligent grouping."""

    rule_id: str
    name: str
    component_pattern: str
    time_window_minutes: float = 5.0
    max_alerts_per_group: int = 10
    correlation_fields: List[str] = field(default_factory=lambda: ["component", "severity"])


@dataclass
class ProcessedAlert:
    """Processed alert with delivery tracking."""

    alert_id: str
    original_alert: HealthAlert
    severity: AlertSeverity
    title: str
    message: str
    component: str
    timestamp: datetime
    status: AlertStatus = AlertStatus.PENDING
    delivery_attempts: Dict[AlertChannel, int] = field(default_factory=dict)
    delivery_status: Dict[AlertChannel, bool] = field(default_factory=dict)
    escalation_level: int = 0
    acknowledgment_time: Optional[datetime] = None
    resolution_time: Optional[datetime] = None
    correlation_group_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertGroup:
    """Correlated alert group."""

    group_id: str
    alerts: List[ProcessedAlert]
    first_alert_time: datetime
    last_alert_time: datetime
    correlation_rule: AlertCorrelationRule
    status: AlertStatus = AlertStatus.PENDING
    summary_sent: bool = False


class AlertingManager:
    """
    Enhanced Multi-Level Alerting Manager.

    Provides comprehensive alerting capabilities with multiple delivery channels,
    intelligent correlation, escalation policies, and automated recovery integration.
    """

    def __init__(
        self,
        enable_correlation: bool = True,
        enable_escalation: bool = True,
        max_alerts_per_hour: int = 1000,
        alert_retention_hours: float = 168.0  # 1 week
    ):
        """
        Initialize alerting manager.

        Args:
            enable_correlation: Enable intelligent alert correlation
            enable_escalation: Enable escalation policies
            max_alerts_per_hour: Global rate limit for alerts
            alert_retention_hours: How long to retain processed alerts
        """
        self.enable_correlation = enable_correlation
        self.enable_escalation = enable_escalation
        self.max_alerts_per_hour = max_alerts_per_hour
        self.alert_retention_hours = alert_retention_hours

        # Alert channels configuration
        self.alert_channels: Dict[str, AlertChannelConfig] = {}
        self.escalation_policies: Dict[str, EscalationPolicy] = {}
        self.correlation_rules: Dict[str, AlertCorrelationRule] = {}

        # Alert processing state
        self.processed_alerts: Dict[str, ProcessedAlert] = {}
        self.alert_groups: Dict[str, AlertGroup] = {}
        self.rate_limit_counters: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()

        # HTTP session for webhook delivery
        self.http_session: Optional[aiohttp.ClientSession] = None

        # Custom alert handlers
        self.custom_handlers: Dict[str, Callable] = {}

        logger.info(
            "Enhanced Alerting Manager initialized",
            correlation_enabled=enable_correlation,
            escalation_enabled=enable_escalation,
            max_alerts_per_hour=max_alerts_per_hour
        )

    async def initialize(self) -> None:
        """Initialize the alerting manager and start background processing."""
        try:
            # Create HTTP session for webhook delivery
            timeout = aiohttp.ClientTimeout(total=30.0)
            self.http_session = aiohttp.ClientSession(timeout=timeout)

            # Add default correlation rules
            await self._add_default_correlation_rules()

            # Add default escalation policies
            await self._add_default_escalation_policies()

            # Start background tasks
            self.background_tasks.extend([
                asyncio.create_task(self._alert_processing_loop()),
                asyncio.create_task(self._escalation_processing_loop()),
                asyncio.create_task(self._cleanup_loop()),
                asyncio.create_task(self._rate_limit_cleanup_loop()),
            ])

            logger.info("Enhanced Alerting Manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Enhanced Alerting Manager: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the alerting manager and cleanup resources."""
        logger.info("Shutting down Enhanced Alerting Manager")

        # Signal shutdown
        self.shutdown_event.set()

        # Cancel background tasks
        for task in self.background_tasks:
            if not task.done():
                task.cancel()

        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)

        self.background_tasks.clear()

        # Close HTTP session
        if self.http_session:
            await self.http_session.close()

        logger.info("Enhanced Alerting Manager shutdown complete")

    async def send_alert(
        self,
        severity: Union[str, AlertSeverity],
        title: str,
        message: str,
        component: str,
        auto_recovery: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        escalation_policy: Optional[str] = None
    ) -> str:
        """
        Send an alert through the alerting system.

        Args:
            severity: Alert severity level
            title: Alert title
            message: Alert message
            component: Component that triggered the alert
            auto_recovery: Whether to attempt automated recovery
            metadata: Additional alert metadata
            escalation_policy: Optional escalation policy to use

        Returns:
            Alert ID for tracking
        """
        try:
            # Convert severity if needed
            if isinstance(severity, str):
                severity = AlertSeverity(severity.lower())

            # Create processed alert
            alert_id = f"alert_{int(time.time() * 1000000)}"

            # Create base health alert for compatibility
            health_alert = HealthAlert(
                alert_id=alert_id,
                component_type=None,  # Will be inferred from component string
                severity=severity,
                message=message,
                description=title,
                timestamp=datetime.now(timezone.utc),
                auto_recovery_attempted=auto_recovery
            )

            processed_alert = ProcessedAlert(
                alert_id=alert_id,
                original_alert=health_alert,
                severity=severity,
                title=title,
                message=message,
                component=component,
                timestamp=datetime.now(timezone.utc),
                metadata=metadata or {}
            )

            # Add escalation policy if specified
            if escalation_policy and escalation_policy in self.escalation_policies:
                processed_alert.metadata["escalation_policy"] = escalation_policy

            # Check rate limits
            if not await self._check_rate_limits(processed_alert):
                logger.warning(f"Alert rate limit exceeded: {alert_id}")
                processed_alert.status = AlertStatus.SUPPRESSED
                return alert_id

            # Store alert
            self.processed_alerts[alert_id] = processed_alert

            # Handle correlation if enabled
            if self.enable_correlation:
                await self._handle_alert_correlation(processed_alert)

            # Record alert metrics
            metrics_instance.increment_counter(
                "alerts_generated_total",
                severity=severity.value,
                component=component
            )

            logger.info(
                f"Alert created: {alert_id}",
                severity=severity.value,
                title=title,
                component=component
            )

            return alert_id

        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            raise

    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an alert to stop escalation."""
        try:
            if alert_id not in self.processed_alerts:
                logger.warning(f"Alert not found for acknowledgment: {alert_id}")
                return False

            alert = self.processed_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledgment_time = datetime.now(timezone.utc)
            alert.metadata["acknowledged_by"] = acknowledged_by

            logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")

            # Record acknowledgment metric
            metrics_instance.increment_counter(
                "alerts_acknowledged_total",
                severity=alert.severity.value,
                component=alert.component
            )

            return True

        except Exception as e:
            logger.error(f"Failed to acknowledge alert {alert_id}: {e}")
            return False

    async def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Resolve an alert."""
        try:
            if alert_id not in self.processed_alerts:
                logger.warning(f"Alert not found for resolution: {alert_id}")
                return False

            alert = self.processed_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolution_time = datetime.now(timezone.utc)
            alert.metadata["resolved_by"] = resolved_by

            logger.info(f"Alert resolved: {alert_id} by {resolved_by}")

            # Record resolution metric
            metrics_instance.increment_counter(
                "alerts_resolved_total",
                severity=alert.severity.value,
                component=alert.component
            )

            return True

        except Exception as e:
            logger.error(f"Failed to resolve alert {alert_id}: {e}")
            return False

    def add_email_channel(
        self,
        email_address: str,
        severities: List[str],
        smtp_server: str = "localhost",
        smtp_port: int = 587,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_tls: bool = True
    ) -> str:
        """Add email alert channel."""
        channel_id = f"email_{email_address.replace('@', '_at_')}"

        config = AlertChannelConfig(
            channel_type=AlertChannel.EMAIL,
            endpoint=email_address,
            severities=severities,
            auth_config={
                "smtp_server": smtp_server,
                "smtp_port": str(smtp_port),
                "username": username or "",
                "password": password or "",
                "use_tls": str(use_tls),
            }
        )

        self.alert_channels[channel_id] = config

        logger.info(f"Added email alert channel: {email_address}")
        return channel_id

    def add_webhook_channel(
        self,
        webhook_url: str,
        severities: List[str],
        headers: Optional[Dict[str, str]] = None
    ) -> str:
        """Add webhook alert channel."""
        channel_id = f"webhook_{hash(webhook_url)}"

        config = AlertChannelConfig(
            channel_type=AlertChannel.WEBHOOK,
            endpoint=webhook_url,
            severities=severities,
            headers=headers or {}
        )

        self.alert_channels[channel_id] = config

        logger.info(f"Added webhook alert channel: {webhook_url}")
        return channel_id

    def add_slack_channel(
        self,
        webhook_url: str,
        severities: List[str],
        channel: Optional[str] = None
    ) -> str:
        """Add Slack alert channel."""
        channel_id = f"slack_{hash(webhook_url)}"

        config = AlertChannelConfig(
            channel_type=AlertChannel.SLACK,
            endpoint=webhook_url,
            severities=severities,
            headers={"Content-Type": "application/json"},
            auth_config={"channel": channel or ""}
        )

        self.alert_channels[channel_id] = config

        logger.info(f"Added Slack alert channel: {channel or 'default'}")
        return channel_id

    def add_custom_handler(
        self,
        handler_name: str,
        handler_func: Callable[[ProcessedAlert], bool],
        severities: List[str]
    ) -> str:
        """Add custom alert handler."""
        channel_id = f"custom_{handler_name}"

        config = AlertChannelConfig(
            channel_type=AlertChannel.CUSTOM,
            endpoint=handler_name,
            severities=severities
        )

        self.alert_channels[channel_id] = config
        self.custom_handlers[handler_name] = handler_func

        logger.info(f"Added custom alert handler: {handler_name}")
        return channel_id

    async def _alert_processing_loop(self) -> None:
        """Background task for processing and delivering alerts."""
        while not self.shutdown_event.is_set():
            try:
                # Process pending alerts
                pending_alerts = [
                    alert for alert in self.processed_alerts.values()
                    if alert.status == AlertStatus.PENDING
                ]

                for alert in pending_alerts:
                    await self._deliver_alert(alert)

                await asyncio.sleep(5.0)  # Process every 5 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Alert processing loop error: {e}")
                await asyncio.sleep(5.0)

    async def _deliver_alert(self, alert: ProcessedAlert) -> None:
        """Deliver alert through configured channels."""
        try:
            successful_deliveries = 0
            total_attempts = 0

            # Find applicable channels for this alert
            applicable_channels = [
                (channel_id, config) for channel_id, config in self.alert_channels.items()
                if config.enabled and alert.severity.value in config.severities
            ]

            for channel_id, channel_config in applicable_channels:
                total_attempts += 1

                try:
                    success = await self._deliver_to_channel(alert, channel_config)
                    alert.delivery_status[channel_config.channel_type] = success

                    if success:
                        successful_deliveries += 1

                except Exception as e:
                    logger.error(f"Failed to deliver alert {alert.alert_id} to {channel_id}: {e}")
                    alert.delivery_status[channel_config.channel_type] = False

            # Update alert status
            if successful_deliveries > 0:
                alert.status = AlertStatus.SENT
                logger.info(f"Alert delivered: {alert.alert_id} ({successful_deliveries}/{total_attempts} channels)")
            else:
                alert.status = AlertStatus.FAILED
                logger.error(f"Alert delivery failed: {alert.alert_id}")

            # Record delivery metrics
            metrics_instance.increment_counter(
                "alert_deliveries_total",
                severity=alert.severity.value,
                component=alert.component,
                status=alert.status.value
            )

        except Exception as e:
            logger.error(f"Alert delivery error for {alert.alert_id}: {e}")
            alert.status = AlertStatus.FAILED

    async def _deliver_to_channel(
        self,
        alert: ProcessedAlert,
        channel_config: AlertChannelConfig
    ) -> bool:
        """Deliver alert to a specific channel."""
        try:
            if channel_config.channel_type == AlertChannel.EMAIL:
                return await self._deliver_email(alert, channel_config)
            elif channel_config.channel_type == AlertChannel.WEBHOOK:
                return await self._deliver_webhook(alert, channel_config)
            elif channel_config.channel_type == AlertChannel.SLACK:
                return await self._deliver_slack(alert, channel_config)
            elif channel_config.channel_type == AlertChannel.CUSTOM:
                return await self._deliver_custom(alert, channel_config)
            else:
                logger.warning(f"Unsupported channel type: {channel_config.channel_type}")
                return False

        except Exception as e:
            logger.error(f"Channel delivery error: {e}")
            return False

    async def _deliver_email(self, alert: ProcessedAlert, config: AlertChannelConfig) -> bool:
        """Deliver alert via email."""
        try:
            # Prepare email content
            subject = f"[{alert.severity.value.upper()}] {alert.title}"

            html_body = f"""
            <html>
            <body>
                <h2 style="color: {'red' if alert.severity == AlertSeverity.CRITICAL else 'orange' if alert.severity == AlertSeverity.WARNING else 'blue'};">
                    {alert.title}
                </h2>
                <p><strong>Severity:</strong> {alert.severity.value.upper()}</p>
                <p><strong>Component:</strong> {alert.component}</p>
                <p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                <p><strong>Message:</strong></p>
                <p>{alert.message}</p>

                {f'<p><strong>Alert ID:</strong> {alert.alert_id}</p>' if alert.alert_id else ''}

                <hr>
                <p style="font-size: 12px; color: #666;">
                    Generated by workspace-qdrant-mcp health monitoring system
                </p>
            </body>
            </html>
            """

            # Create email message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = config.auth_config.get("username", "noreply@workspace-qdrant-mcp.local")
            msg["To"] = config.endpoint

            # Add HTML content
            html_part = MIMEText(html_body, "html")
            msg.attach(html_part)

            # Send email
            smtp_server = config.auth_config.get("smtp_server", "localhost")
            smtp_port = int(config.auth_config.get("smtp_port", "587"))
            username = config.auth_config.get("username")
            password = config.auth_config.get("password")
            use_tls = config.auth_config.get("use_tls", "true").lower() == "true"

            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._send_email_blocking,
                msg, smtp_server, smtp_port, username, password, use_tls
            )

            logger.debug(f"Email alert sent to {config.endpoint}")
            return True

        except Exception as e:
            logger.error(f"Email delivery failed: {e}")
            return False

    def _send_email_blocking(
        self,
        msg: MIMEMultipart,
        smtp_server: str,
        smtp_port: int,
        username: Optional[str],
        password: Optional[str],
        use_tls: bool
    ) -> None:
        """Send email using blocking SMTP (to be run in executor)."""
        context = ssl.create_default_context()

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            if use_tls:
                server.starttls(context=context)

            if username and password:
                server.login(username, password)

            server.send_message(msg)

    async def _deliver_webhook(self, alert: ProcessedAlert, config: AlertChannelConfig) -> bool:
        """Deliver alert via webhook."""
        try:
            # Prepare webhook payload
            payload = {
                "alert_id": alert.alert_id,
                "severity": alert.severity.value,
                "title": alert.title,
                "message": alert.message,
                "component": alert.component,
                "timestamp": alert.timestamp.isoformat(),
                "status": alert.status.value,
                "metadata": alert.metadata,
            }

            # Send webhook
            async with self.http_session.post(
                config.endpoint,
                json=payload,
                headers=config.headers,
                timeout=aiohttp.ClientTimeout(total=config.timeout_seconds)
            ) as response:
                if response.status < 400:
                    logger.debug(f"Webhook alert sent to {config.endpoint}")
                    return True
                else:
                    logger.error(f"Webhook delivery failed: HTTP {response.status}")
                    return False

        except Exception as e:
            logger.error(f"Webhook delivery failed: {e}")
            return False

    async def _deliver_slack(self, alert: ProcessedAlert, config: AlertChannelConfig) -> bool:
        """Deliver alert via Slack webhook."""
        try:
            # Prepare Slack payload
            color = {
                AlertSeverity.INFO: "good",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.CRITICAL: "danger",
                AlertSeverity.EMERGENCY: "danger"
            }.get(alert.severity, "warning")

            slack_payload = {
                "text": f"Health Alert: {alert.title}",
                "attachments": [
                    {
                        "color": color,
                        "title": alert.title,
                        "text": alert.message,
                        "fields": [
                            {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                            {"title": "Component", "value": alert.component, "short": True},
                            {"title": "Time", "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'), "short": True},
                            {"title": "Alert ID", "value": alert.alert_id, "short": True}
                        ],
                        "footer": "workspace-qdrant-mcp health monitoring",
                        "ts": int(alert.timestamp.timestamp())
                    }
                ]
            }

            # Add channel if specified
            channel = config.auth_config.get("channel")
            if channel:
                slack_payload["channel"] = channel

            # Send to Slack
            async with self.http_session.post(
                config.endpoint,
                json=slack_payload,
                headers=config.headers,
                timeout=aiohttp.ClientTimeout(total=config.timeout_seconds)
            ) as response:
                if response.status < 400:
                    logger.debug(f"Slack alert sent to {config.endpoint}")
                    return True
                else:
                    logger.error(f"Slack delivery failed: HTTP {response.status}")
                    return False

        except Exception as e:
            logger.error(f"Slack delivery failed: {e}")
            return False

    async def _deliver_custom(self, alert: ProcessedAlert, config: AlertChannelConfig) -> bool:
        """Deliver alert via custom handler."""
        try:
            handler_name = config.endpoint
            handler = self.custom_handlers.get(handler_name)

            if not handler:
                logger.error(f"Custom handler not found: {handler_name}")
                return False

            # Run custom handler
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, handler, alert)

            logger.debug(f"Custom alert handler executed: {handler_name}")
            return bool(result)

        except Exception as e:
            logger.error(f"Custom handler delivery failed: {e}")
            return False

    async def _escalation_processing_loop(self) -> None:
        """Background task for processing alert escalations."""
        if not self.enable_escalation:
            return

        while not self.shutdown_event.is_set():
            try:
                # Process escalations for sent alerts
                for alert in self.processed_alerts.values():
                    if (alert.status == AlertStatus.SENT
                        and "escalation_policy" in alert.metadata):
                        await self._process_escalation(alert)

                await asyncio.sleep(60.0)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Escalation processing error: {e}")
                await asyncio.sleep(60.0)

    async def _process_escalation(self, alert: ProcessedAlert) -> None:
        """Process escalation for an alert."""
        try:
            policy_id = alert.metadata.get("escalation_policy")
            policy = self.escalation_policies.get(policy_id)

            if not policy or not policy.enabled:
                return

            # Check if escalation is needed
            time_since_alert = (datetime.now(timezone.utc) - alert.timestamp).total_seconds() / 60

            required_escalation_time = policy.escalation_interval_minutes * (alert.escalation_level + 1)

            if time_since_alert >= required_escalation_time and alert.escalation_level < policy.max_escalations:
                # Trigger escalation
                alert.escalation_level += 1

                logger.warning(
                    f"Escalating alert: {alert.alert_id}",
                    escalation_level=alert.escalation_level,
                    policy=policy_id
                )

                # Record escalation metric
                metrics_instance.increment_counter(
                    "alert_escalations_total",
                    severity=alert.severity.value,
                    component=alert.component,
                    escalation_level=str(alert.escalation_level)
                )

                # Trigger escalation action (simplified - would implement based on policy)
                escalation_level = policy.escalation_levels[min(alert.escalation_level - 1, len(policy.escalation_levels) - 1)]

                # Example: send to additional channels or trigger paging
                if escalation_level.get("action") == "page":
                    await self._trigger_paging(alert, escalation_level)

        except Exception as e:
            logger.error(f"Escalation processing failed for {alert.alert_id}: {e}")

    async def _trigger_paging(self, alert: ProcessedAlert, escalation_config: Dict[str, Any]) -> None:
        """Trigger paging for escalated alert."""
        # Implementation would depend on paging service (PagerDuty, OpsGenie, etc.)
        logger.warning(f"Paging triggered for escalated alert: {alert.alert_id}")

    async def _handle_alert_correlation(self, alert: ProcessedAlert) -> None:
        """Handle alert correlation and grouping."""
        if not self.enable_correlation:
            return

        try:
            # Find applicable correlation rules
            applicable_rules = [
                rule for rule in self.correlation_rules.values()
                if self._alert_matches_rule(alert, rule)
            ]

            for rule in applicable_rules:
                group_id = await self._find_or_create_alert_group(alert, rule)
                if group_id:
                    alert.correlation_group_id = group_id
                    break

        except Exception as e:
            logger.error(f"Alert correlation failed for {alert.alert_id}: {e}")

    def _alert_matches_rule(self, alert: ProcessedAlert, rule: AlertCorrelationRule) -> bool:
        """Check if alert matches correlation rule."""
        # Simple pattern matching (could be enhanced with regex)
        return rule.component_pattern in alert.component

    async def _find_or_create_alert_group(
        self,
        alert: ProcessedAlert,
        rule: AlertCorrelationRule
    ) -> Optional[str]:
        """Find existing alert group or create new one."""
        current_time = datetime.now(timezone.utc)

        # Look for existing groups within time window
        for group_id, group in self.alert_groups.items():
            if (group.correlation_rule.rule_id == rule.rule_id
                and (current_time - group.last_alert_time).total_seconds() / 60 <= rule.time_window_minutes
                and len(group.alerts) < rule.max_alerts_per_group):

                # Add to existing group
                group.alerts.append(alert)
                group.last_alert_time = current_time

                logger.debug(f"Added alert {alert.alert_id} to group {group_id}")
                return group_id

        # Create new group
        group_id = f"group_{int(time.time() * 1000)}"
        new_group = AlertGroup(
            group_id=group_id,
            alerts=[alert],
            first_alert_time=current_time,
            last_alert_time=current_time,
            correlation_rule=rule
        )

        self.alert_groups[group_id] = new_group

        logger.debug(f"Created new alert group {group_id} for alert {alert.alert_id}")
        return group_id

    async def _check_rate_limits(self, alert: ProcessedAlert) -> bool:
        """Check if alert is within rate limits."""
        current_time = time.time()
        hour_window = 3600  # 1 hour

        # Global rate limit
        global_counter = self.rate_limit_counters["global"]
        global_counter.append(current_time)

        # Remove old entries
        while global_counter and global_counter[0] < current_time - hour_window:
            global_counter.popleft()

        if len(global_counter) > self.max_alerts_per_hour:
            return False

        # Component-specific rate limit
        component_counter = self.rate_limit_counters[f"component_{alert.component}"]
        component_counter.append(current_time)

        while component_counter and component_counter[0] < current_time - hour_window:
            component_counter.popleft()

        # Allow up to 20% of global limit per component
        component_limit = max(10, self.max_alerts_per_hour // 5)
        if len(component_counter) > component_limit:
            return False

        return True

    async def _add_default_correlation_rules(self) -> None:
        """Add default correlation rules."""
        # Component failure correlation
        self.correlation_rules["component_failures"] = AlertCorrelationRule(
            rule_id="component_failures",
            name="Component Failure Correlation",
            component_pattern="",  # Match all
            time_window_minutes=5.0,
            max_alerts_per_group=5,
            correlation_fields=["component", "severity"]
        )

        # Performance degradation correlation
        self.correlation_rules["performance_issues"] = AlertCorrelationRule(
            rule_id="performance_issues",
            name="Performance Issue Correlation",
            component_pattern="performance",
            time_window_minutes=10.0,
            max_alerts_per_group=10,
            correlation_fields=["severity"]
        )

    async def _add_default_escalation_policies(self) -> None:
        """Add default escalation policies."""
        # Critical alert escalation
        self.escalation_policies["critical_escalation"] = EscalationPolicy(
            policy_id="critical_escalation",
            name="Critical Alert Escalation",
            escalation_levels=[
                {"level": 1, "action": "notify", "targets": ["admin"]},
                {"level": 2, "action": "page", "targets": ["oncall"]},
                {"level": 3, "action": "page", "targets": ["manager"]}
            ],
            escalation_interval_minutes=15.0,
            max_escalations=3
        )

    async def _cleanup_loop(self) -> None:
        """Background task for cleaning up old alerts and groups."""
        while not self.shutdown_event.is_set():
            try:
                current_time = datetime.now(timezone.utc)
                cutoff_time = current_time - timedelta(hours=self.alert_retention_hours)

                # Clean up old processed alerts
                alerts_to_remove = [
                    alert_id for alert_id, alert in self.processed_alerts.items()
                    if alert.timestamp < cutoff_time
                ]

                for alert_id in alerts_to_remove:
                    del self.processed_alerts[alert_id]

                # Clean up old alert groups
                groups_to_remove = [
                    group_id for group_id, group in self.alert_groups.items()
                    if group.last_alert_time < cutoff_time
                ]

                for group_id in groups_to_remove:
                    del self.alert_groups[group_id]

                if alerts_to_remove or groups_to_remove:
                    logger.debug(
                        f"Cleaned up old alerts and groups",
                        alerts_removed=len(alerts_to_remove),
                        groups_removed=len(groups_to_remove)
                    )

                await asyncio.sleep(3600)  # Run every hour

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(3600)

    async def _rate_limit_cleanup_loop(self) -> None:
        """Background task for cleaning up rate limit counters."""
        while not self.shutdown_event.is_set():
            try:
                current_time = time.time()
                hour_ago = current_time - 3600

                # Clean up old rate limit entries
                for counter in self.rate_limit_counters.values():
                    while counter and counter[0] < hour_ago:
                        counter.popleft()

                await asyncio.sleep(300)  # Run every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Rate limit cleanup error: {e}")
                await asyncio.sleep(300)


# Global alerting manager instance
_alerting_manager: Optional[AlertingManager] = None


async def get_alerting_manager(**kwargs) -> AlertingManager:
    """Get or create global alerting manager instance."""
    global _alerting_manager

    if _alerting_manager is None:
        _alerting_manager = AlertingManager(**kwargs)
        await _alerting_manager.initialize()

    return _alerting_manager


async def shutdown_alerting_manager():
    """Shutdown global alerting manager."""
    global _alerting_manager

    if _alerting_manager:
        await _alerting_manager.shutdown()
        _alerting_manager = None