"""
Comprehensive unit tests for security monitoring system.

Tests cover all components with edge cases and error conditions:
- SecurityMonitor integration and lifecycle
- SecurityMetrics collection and aggregation
- AlertingSystem rule evaluation and notifications
- SecurityEventLogger functionality
- Error handling and resource management
- Concurrent operations and thread safety
"""

import asyncio
import json
import tempfile
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock, mock_open, patch

import pytest

from src.python.common.security.security_monitor import (
    AlertingSystem,
    AlertLevel,
    MetricType,
    SecurityAlert,
    SecurityEventLogger,
    SecurityMetric,
    SecurityMetrics,
    SecurityMonitor,
)
from src.python.common.security.threat_detection import (
    SecurityEvent,
    ThreatDetection,
    ThreatLevel,
    ThreatType,
)


class TestSecurityMetrics:
    """Test security metrics collection and analysis."""

    @pytest.fixture
    def metrics(self):
        """Create SecurityMetrics instance for testing."""
        return SecurityMetrics(retention_period=timedelta(hours=1))

    @pytest.fixture
    def sample_metric(self):
        """Create sample security metric."""
        return SecurityMetric(
            metric_type=MetricType.THREAT_COUNT,
            value=5,
            timestamp=datetime.utcnow(),
            tags={'source': 'test'},
            metadata={'test': 'data'}
        )

    @pytest.mark.asyncio
    async def test_record_metric(self, metrics, sample_metric):
        """Test basic metric recording."""
        await metrics.record_metric(sample_metric)

        assert len(metrics.metrics[MetricType.THREAT_COUNT]) == 1
        recorded_metric = metrics.metrics[MetricType.THREAT_COUNT][0]
        assert recorded_metric.value == 5
        assert recorded_metric.tags == {'source': 'test'}

    @pytest.mark.asyncio
    async def test_metric_summary_empty(self, metrics):
        """Test metric summary with no data."""
        summary = await metrics.get_metric_summary(MetricType.THREAT_COUNT)

        assert summary['count'] == 0
        assert summary['total'] == 0
        assert summary['average'] == 0
        assert summary['trend'] == 'stable'

    @pytest.mark.asyncio
    async def test_metric_summary_with_data(self, metrics):
        """Test metric summary with recorded data."""
        # Record multiple metrics
        values = [1, 3, 5, 7, 9]
        for value in values:
            metric = SecurityMetric(
                metric_type=MetricType.THREAT_COUNT,
                value=value,
                timestamp=datetime.utcnow()
            )
            await metrics.record_metric(metric)

        summary = await metrics.get_metric_summary(MetricType.THREAT_COUNT)

        assert summary['count'] == 5
        assert summary['total'] == 25
        assert summary['average'] == 5.0
        assert summary['min'] == 1
        assert summary['max'] == 9
        assert summary['std_dev'] > 0

    @pytest.mark.asyncio
    async def test_metric_trend_calculation(self, metrics):
        """Test trend calculation for metrics."""
        # Increasing trend
        for i in range(10):
            metric = SecurityMetric(
                metric_type=MetricType.THREAT_COUNT,
                value=i,  # 0, 1, 2, ..., 9
                timestamp=datetime.utcnow()
            )
            await metrics.record_metric(metric)

        summary = await metrics.get_metric_summary(MetricType.THREAT_COUNT)
        assert summary['trend'] == 'increasing'

        # Decreasing trend
        metrics.metrics[MetricType.AUTH_FAILURES] = deque(maxlen=10000)
        for i in range(10, 0, -1):
            metric = SecurityMetric(
                metric_type=MetricType.AUTH_FAILURES,
                value=i,  # 10, 9, 8, ..., 1
                timestamp=datetime.utcnow()
            )
            await metrics.record_metric(metric)

        summary = await metrics.get_metric_summary(MetricType.AUTH_FAILURES)
        assert summary['trend'] == 'decreasing'

    @pytest.mark.asyncio
    async def test_metric_time_window_filtering(self, metrics):
        """Test metric filtering by time window."""
        old_time = datetime.utcnow() - timedelta(hours=2)
        recent_time = datetime.utcnow()

        # Add old metric
        old_metric = SecurityMetric(
            metric_type=MetricType.THREAT_COUNT,
            value=10,
            timestamp=old_time
        )
        await metrics.record_metric(old_metric)

        # Add recent metric
        recent_metric = SecurityMetric(
            metric_type=MetricType.THREAT_COUNT,
            value=20,
            timestamp=recent_time
        )
        await metrics.record_metric(recent_metric)

        # Get summary for last hour only
        summary = await metrics.get_metric_summary(
            MetricType.THREAT_COUNT,
            time_window=timedelta(hours=1)
        )

        # Should only include recent metric
        assert summary['count'] == 1
        assert summary['total'] == 20

    @pytest.mark.asyncio
    async def test_all_metric_summaries(self, metrics):
        """Test getting summaries for all metric types."""
        # Add metrics for different types
        for metric_type in [MetricType.THREAT_COUNT, MetricType.AUTH_FAILURES]:
            metric = SecurityMetric(
                metric_type=metric_type,
                value=5,
                timestamp=datetime.utcnow()
            )
            await metrics.record_metric(metric)

        summaries = await metrics.get_all_metric_summaries()

        assert len(summaries) == len(MetricType)
        assert 'threat_count' in summaries
        assert 'auth_failures' in summaries

        # Metrics with data should have count > 0
        assert summaries['threat_count']['count'] == 1
        assert summaries['auth_failures']['count'] == 1

        # Metrics without data should have count = 0
        assert summaries['rate_violations']['count'] == 0

    @pytest.mark.asyncio
    async def test_metric_cleanup(self, metrics):
        """Test cleanup of old metrics."""
        # Set short retention period for testing
        metrics.retention_period = timedelta(minutes=1)

        # Add old metric
        old_metric = SecurityMetric(
            metric_type=MetricType.THREAT_COUNT,
            value=10,
            timestamp=datetime.utcnow() - timedelta(minutes=2)
        )
        await metrics.record_metric(old_metric)

        # Add recent metric
        recent_metric = SecurityMetric(
            metric_type=MetricType.THREAT_COUNT,
            value=20,
            timestamp=datetime.utcnow()
        )
        await metrics.record_metric(recent_metric)

        # Cleanup should remove old metric
        assert len(metrics.metrics[MetricType.THREAT_COUNT]) == 1
        assert metrics.metrics[MetricType.THREAT_COUNT][0].value == 20

    @pytest.mark.asyncio
    async def test_concurrent_metric_recording(self, metrics):
        """Test concurrent metric recording."""
        # Create multiple tasks recording metrics simultaneously
        tasks = []
        for i in range(10):
            metric = SecurityMetric(
                metric_type=MetricType.THREAT_COUNT,
                value=i,
                timestamp=datetime.utcnow()
            )
            tasks.append(metrics.record_metric(metric))

        # Execute all tasks concurrently
        await asyncio.gather(*tasks)

        # All metrics should be recorded
        assert len(metrics.metrics[MetricType.THREAT_COUNT]) == 10

    def test_security_metric_to_dict(self, sample_metric):
        """Test SecurityMetric to_dict conversion."""
        metric_dict = sample_metric.to_dict()

        assert metric_dict['metric_type'] == 'threat_count'
        assert metric_dict['value'] == 5
        assert 'timestamp' in metric_dict
        assert metric_dict['tags'] == {'source': 'test'}
        assert metric_dict['metadata'] == {'test': 'data'}


class TestAlertingSystem:
    """Test alerting system functionality."""

    @pytest.fixture
    def alerting(self):
        """Create AlertingSystem instance for testing."""
        return AlertingSystem()

    @pytest.fixture
    def sample_threat(self):
        """Create sample threat detection."""
        return ThreatDetection(
            threat_id="test_threat_001",
            threat_type=ThreatType.SQL_INJECTION,
            threat_level=ThreatLevel.HIGH,
            confidence=0.85,
            description="SQL injection detected in query",
            source_events=[],
            mitigation_suggestions=["Sanitize inputs", "Use parameterized queries"]
        )

    def test_add_alert_rule(self, alerting):
        """Test adding alert rules."""
        rule = {
            'name': 'High Threat Count',
            'condition': 'threat_count.total > 5',
            'alert_level': 'ERROR',
            'message': 'Too many threats detected'
        }

        alerting.add_alert_rule(rule)
        assert len(alerting.alert_rules) == 1
        assert alerting.alert_rules[0]['name'] == 'High Threat Count'

    def test_add_invalid_alert_rule(self, alerting):
        """Test adding invalid alert rules."""
        invalid_rule = {
            'name': 'Incomplete Rule'
            # Missing required fields
        }

        with pytest.raises(ValueError, match="Alert rule must contain"):
            alerting.add_alert_rule(invalid_rule)

    def test_register_notification_handler(self, alerting):
        """Test registering notification handlers."""
        def test_handler(alert):
            pass

        alerting.register_notification_handler(test_handler)
        assert len(alerting.notification_handlers) == 1

    @pytest.mark.asyncio
    async def test_create_threat_alert(self, alerting, sample_threat):
        """Test creating alerts from threat detections."""
        alert = await alerting._create_threat_alert(sample_threat)

        assert alert is not None
        assert alert.alert_level == AlertLevel.ERROR  # HIGH threat -> ERROR alert
        assert "SQL Injection" in alert.title
        assert alert.source == "threat_detection"
        assert alert.metadata['threat_type'] == 'sql_injection'
        assert alert.metadata['confidence'] == 0.85

    @pytest.mark.asyncio
    async def test_evaluate_metric_rule(self, alerting):
        """Test metric-based alert rule evaluation."""
        rule = {
            'name': 'High Error Rate',
            'condition': 'error_rate.total > 10',
            'alert_level': 'WARNING',
            'message': 'High error rate detected'
        }

        # Metrics that should trigger the rule
        metrics = {
            'error_rate': {
                'total': 15,
                'average': 3.0,
                'count': 5
            }
        }

        alert = await alerting._evaluate_metric_rule(rule, metrics)

        assert alert is not None
        assert alert.alert_level == AlertLevel.WARNING
        assert alert.title == 'High Error Rate'
        assert alert.source == "metric_rule"

    @pytest.mark.asyncio
    async def test_evaluate_metric_rule_no_trigger(self, alerting):
        """Test metric rule that doesn't trigger."""
        rule = {
            'name': 'High Error Rate',
            'condition': 'error_rate.total > 20',
            'alert_level': 'WARNING',
            'message': 'High error rate detected'
        }

        metrics = {
            'error_rate': {
                'total': 10,  # Below threshold
                'average': 2.0,
                'count': 5
            }
        }

        alert = await alerting._evaluate_metric_rule(rule, metrics)
        assert alert is None

    def test_evaluate_condition(self, alerting):
        """Test condition evaluation logic."""
        metrics = {
            'threat_count': {
                'total': 15,
                'average': 3.0,
                'max': 5
            },
            'error_rate': {
                'total': 8
            }
        }

        # Test various operators
        assert alerting._evaluate_condition("threat_count.total > 10", metrics)
        assert not alerting._evaluate_condition("threat_count.total < 10", metrics)
        assert alerting._evaluate_condition("threat_count.average >= 3.0", metrics)
        assert alerting._evaluate_condition("error_rate.total <= 8", metrics)
        assert alerting._evaluate_condition("threat_count.max == 5", metrics)
        assert alerting._evaluate_condition("threat_count.max != 3", metrics)

        # Test invalid conditions
        assert not alerting._evaluate_condition("invalid.path > 5", metrics)
        assert not alerting._evaluate_condition("malformed condition", metrics)

    def test_alert_suppression(self, alerting):
        """Test alert suppression functionality."""
        alert = SecurityAlert(
            alert_id="test_001",
            alert_level=AlertLevel.WARNING,
            title="Test Alert",
            description="Test description",
            source="test"
        )

        # First alert should not be suppressed
        assert not alerting._is_suppressed(alert)

        # Mark as sent
        suppression_key = f"{alert.source}_{alert.title}"
        alerting.suppressed_alerts[suppression_key] = datetime.utcnow()

        # Second alert should be suppressed
        assert alerting._is_suppressed(alert)

        # After suppression period, should not be suppressed
        alerting.suppressed_alerts[suppression_key] = (
            datetime.utcnow() - timedelta(minutes=15)
        )
        assert not alerting._is_suppressed(alert)

    @pytest.mark.asyncio
    async def test_evaluate_alert_rules_integration(self, alerting, sample_threat):
        """Test complete alert rule evaluation."""
        # Add a rule
        rule = {
            'name': 'Threat Detection',
            'condition': 'threat_count.total > 0',
            'alert_level': 'ERROR',
            'message': 'Threats detected'
        }
        alerting.add_alert_rule(rule)

        metrics = {
            'threat_count': {'total': 5, 'average': 1.0, 'max': 2}
        }

        threats = [sample_threat]

        alerts = await alerting.evaluate_alert_rules(metrics, threats)

        # Should generate alerts for both the threat and the metric rule
        assert len(alerts) >= 1
        alert_sources = {alert.source for alert in alerts}
        expected_sources = {"threat_detection", "metric_rule"}
        assert alert_sources.intersection(expected_sources)

    @pytest.mark.asyncio
    @patch('smtplib.SMTP')
    async def test_send_email_notification(self, mock_smtp, alerting):
        """Test email notification sending."""
        # Configure email settings
        alerting.email_config = {
            'enabled': True,
            'smtp_server': 'smtp.example.com',
            'smtp_port': 587,
            'username': 'security@example.com',
            'password': 'password',
            'recipients': ['admin@example.com'],
            'min_level': 'ERROR'
        }

        alert = SecurityAlert(
            alert_id="email_test",
            alert_level=AlertLevel.ERROR,
            title="Test Alert",
            description="Test email alert",
            source="test"
        )

        # Mock SMTP server
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        await alerting._send_email_notification(alert)

        # Verify SMTP was called
        mock_smtp.assert_called_once_with('smtp.example.com', 587)
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once()
        mock_server.send_message.assert_called_once()

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.post')
    async def test_send_webhook_notification(self, mock_post, alerting):
        """Test webhook notification sending."""
        # Configure webhook settings
        alerting.webhook_config = {
            'urls': ['http://webhook1.example.com', 'http://webhook2.example.com']
        }

        alert = SecurityAlert(
            alert_id="webhook_test",
            alert_level=AlertLevel.WARNING,
            title="Test Alert",
            description="Test webhook alert",
            source="test"
        )

        # Mock successful webhook response
        mock_response = Mock()
        mock_response.status = 200
        mock_post.return_value.__aenter__.return_value = mock_response

        await alerting._send_webhook_notification(alert)

        # Should have called webhook endpoints
        assert mock_post.call_count == 2

    def test_acknowledge_alert(self, alerting):
        """Test alert acknowledgment."""
        alert = SecurityAlert(
            alert_id="ack_test",
            alert_level=AlertLevel.WARNING,
            title="Test Alert",
            description="Test acknowledgment",
            source="test"
        )

        # Add alert to system
        alerting.alerts[alert.alert_id] = alert

        # Acknowledge alert
        result = alerting.acknowledge_alert(alert.alert_id, "test_user")

        assert result
        assert alert.acknowledged
        assert alert.metadata['acknowledged_by'] == "test_user"
        assert 'acknowledged_at' in alert.metadata

        # Try to acknowledge non-existent alert
        result = alerting.acknowledge_alert("non_existent", "test_user")
        assert not result

    def test_get_active_alerts(self, alerting):
        """Test getting active (unacknowledged) alerts."""
        # Add various alerts
        alert1 = SecurityAlert(
            alert_id="alert1",
            alert_level=AlertLevel.ERROR,
            title="Error Alert",
            description="Test",
            source="test"
        )

        alert2 = SecurityAlert(
            alert_id="alert2",
            alert_level=AlertLevel.WARNING,
            title="Warning Alert",
            description="Test",
            source="test"
        )

        alert3 = SecurityAlert(
            alert_id="alert3",
            alert_level=AlertLevel.ERROR,
            title="Acknowledged Alert",
            description="Test",
            source="test",
            acknowledged=True
        )

        alerting.alerts["alert1"] = alert1
        alerting.alerts["alert2"] = alert2
        alerting.alerts["alert3"] = alert3

        # Get all active alerts
        active = alerting.get_active_alerts()
        assert len(active) == 2  # alert3 is acknowledged

        # Get active alerts by level
        error_alerts = alerting.get_active_alerts(alert_level=AlertLevel.ERROR)
        assert len(error_alerts) == 1
        assert error_alerts[0].alert_id == "alert1"

    def test_security_alert_to_dict(self):
        """Test SecurityAlert to_dict conversion."""
        alert = SecurityAlert(
            alert_id="dict_test",
            alert_level=AlertLevel.CRITICAL,
            title="Test Alert",
            description="Test description",
            source="test",
            metadata={'key': 'value'},
            acknowledged=True
        )

        alert_dict = alert.to_dict()

        assert alert_dict['alert_id'] == "dict_test"
        assert alert_dict['alert_level'] == "CRITICAL"
        assert alert_dict['title'] == "Test Alert"
        assert alert_dict['acknowledged']
        assert alert_dict['metadata'] == {'key': 'value'}


class TestSecurityEventLogger:
    """Test security event logging functionality."""

    @pytest.fixture
    def temp_log_file(self):
        """Create temporary log file for testing."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.log')
        temp_path = Path(temp_file.name)
        temp_file.close()
        yield temp_path
        temp_path.unlink(missing_ok=True)

    @pytest.fixture
    def logger(self, temp_log_file):
        """Create SecurityEventLogger for testing."""
        return SecurityEventLogger(log_file=temp_log_file)

    @pytest.fixture
    def sample_event(self):
        """Create sample security event."""
        return SecurityEvent(
            event_id="log_test_001",
            timestamp=datetime.utcnow(),
            event_type="query",
            source_ip="192.168.1.100",
            user_id="test_user",
            collection="test_collection",
            query="SELECT * FROM documents"
        )

    @pytest.fixture
    def sample_threat(self):
        """Create sample threat detection."""
        return ThreatDetection(
            threat_id="log_threat_001",
            threat_type=ThreatType.SQL_INJECTION,
            threat_level=ThreatLevel.HIGH,
            confidence=0.85,
            description="SQL injection detected",
            source_events=[],
            mitigation_suggestions=["Sanitize inputs"]
        )

    @pytest.mark.asyncio
    async def test_log_event(self, logger, sample_event, temp_log_file):
        """Test security event logging."""
        await logger.log_event(sample_event)

        # Check buffer
        assert len(logger.event_buffer) == 1
        logged_event = logger.event_buffer[0]
        assert logged_event['event_id'] == "log_test_001"
        assert logged_event['user_id'] == "test_user"

        # Check file writing
        assert temp_log_file.exists()
        content = temp_log_file.read_text()
        assert "log_test_001" in content
        assert "test_user" in content

    @pytest.mark.asyncio
    async def test_log_threat(self, logger, sample_threat, temp_log_file):
        """Test threat logging."""
        await logger.log_threat(sample_threat)

        # Check buffer
        assert len(logger.event_buffer) == 1
        logged_threat = logger.event_buffer[0]
        assert logged_threat['threat_id'] == "log_threat_001"
        assert logged_threat['threat_type'] == 'sql_injection'
        assert logged_threat['confidence'] == 0.85

        # Check file writing
        content = temp_log_file.read_text()
        assert "log_threat_001" in content
        assert "sql_injection" in content

    @pytest.mark.asyncio
    async def test_log_event_with_query_hash(self, logger, sample_event):
        """Test that queries are hashed for privacy."""
        await logger.log_event(sample_event)

        logged_event = logger.event_buffer[0]
        assert 'query_hash' in logged_event
        assert logged_event['query_hash'] is not None
        assert logged_event['query_hash'] != sample_event.query  # Should be hashed

    @pytest.mark.asyncio
    async def test_log_event_without_query(self, logger):
        """Test logging event without query."""
        event = SecurityEvent(
            event_id="no_query",
            timestamp=datetime.utcnow(),
            event_type="connection",
            source_ip="192.168.1.100",
            user_id="test_user",
            collection=None,
            query=None
        )

        await logger.log_event(event)

        logged_event = logger.event_buffer[0]
        assert logged_event['query_hash'] is None

    def test_get_recent_events(self, logger):
        """Test getting recent events from buffer."""
        # Add events to buffer directly for testing
        for i in range(20):
            logger.event_buffer.append({
                'timestamp': datetime.utcnow().isoformat(),
                'event_id': f"event_{i}",
                'test_data': i
            })

        # Get last 10 events
        recent = logger.get_recent_events(limit=10)
        assert len(recent) == 10
        assert recent[-1]['event_id'] == "event_19"  # Most recent
        assert recent[0]['event_id'] == "event_10"   # 10th from end

    @pytest.mark.asyncio
    @patch('builtins.open', side_effect=OSError("File write error"))
    async def test_log_file_write_error(self, mock_open, logger, sample_event):
        """Test handling of file write errors."""
        # Should not crash on file write error
        await logger.log_event(sample_event)

        # Event should still be in buffer
        assert len(logger.event_buffer) == 1

    @pytest.mark.asyncio
    async def test_concurrent_logging(self, logger, temp_log_file):
        """Test concurrent logging operations."""
        events = []
        for i in range(10):
            events.append(SecurityEvent(
                event_id=f"concurrent_{i}",
                timestamp=datetime.utcnow(),
                event_type="query",
                source_ip="192.168.1.100",
                user_id=f"user_{i}",
                collection="test_collection",
                query=f"SELECT * FROM documents WHERE id = {i}"
            ))

        # Log all events concurrently
        tasks = [logger.log_event(event) for event in events]
        await asyncio.gather(*tasks)

        # All events should be logged
        assert len(logger.event_buffer) == 10

        # File should contain all events
        content = temp_log_file.read_text()
        for i in range(10):
            assert f"concurrent_{i}" in content


class TestSecurityMonitor:
    """Test the main security monitor integration."""

    @pytest.fixture
    def temp_log_file(self):
        """Create temporary log file for testing."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.log')
        temp_path = Path(temp_file.name)
        temp_file.close()
        yield temp_path
        temp_path.unlink(missing_ok=True)

    @pytest.fixture
    def monitor_config(self, temp_log_file):
        """Create monitor configuration for testing."""
        return {
            'metric_retention_hours': 1,
            'check_interval_seconds': 1,
            'log_file': str(temp_log_file),
            'alerting': {
                'email': {'enabled': False},
                'webhooks': {'urls': []}
            }
        }

    @pytest.fixture
    def monitor(self, monitor_config):
        """Create SecurityMonitor for testing."""
        return SecurityMonitor(monitor_config)

    @pytest.fixture
    def sample_event(self):
        """Create sample security event."""
        return SecurityEvent(
            event_id="monitor_test_001",
            timestamp=datetime.utcnow(),
            event_type="query",
            source_ip="192.168.1.100",
            user_id="test_user",
            collection="test_collection",
            query="SELECT * FROM documents"
        )

    @pytest.fixture
    def sample_threat(self):
        """Create sample threat detection."""
        return ThreatDetection(
            threat_id="monitor_threat_001",
            threat_type=ThreatType.SQL_INJECTION,
            threat_level=ThreatLevel.HIGH,
            confidence=0.85,
            description="SQL injection detected",
            source_events=[],
            mitigation_suggestions=["Sanitize inputs"]
        )

    @pytest.mark.asyncio
    async def test_record_security_event(self, monitor, sample_event):
        """Test recording security events."""
        await monitor.record_security_event(sample_event)

        # Event should be logged
        assert len(monitor.event_logger.event_buffer) == 1

        # Collection access metric should be recorded
        assert len(monitor.metrics.metrics[MetricType.COLLECTION_ACCESS]) == 1

    @pytest.mark.asyncio
    async def test_record_threat(self, monitor, sample_threat):
        """Test recording threat detections."""
        await monitor.record_threat(sample_threat)

        # Threat should be added to active threats
        assert len(monitor.active_threats) == 1
        assert monitor.active_threats[0].threat_id == "monitor_threat_001"

        # Threat metric should be recorded
        assert len(monitor.metrics.metrics[MetricType.THREAT_COUNT]) == 1

    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, monitor):
        """Test starting and stopping monitoring."""
        # Initially not monitoring
        assert not monitor.monitoring_active

        # Start monitoring
        await monitor.start_monitoring()
        assert monitor.monitoring_active
        assert monitor.monitor_task is not None

        # Give monitoring loop time to run at least once
        await asyncio.sleep(0.1)

        # Stop monitoring
        await monitor.stop_monitoring()
        assert not monitor.monitoring_active

    @pytest.mark.asyncio
    async def test_start_monitoring_already_active(self, monitor):
        """Test starting monitoring when already active."""
        await monitor.start_monitoring()

        # Try to start again - should not create new task
        original_task = monitor.monitor_task
        await monitor.start_monitoring()

        assert monitor.monitor_task is original_task

        await monitor.stop_monitoring()

    @pytest.mark.asyncio
    async def test_monitoring_check_cycle(self, monitor, sample_threat):
        """Test monitoring check cycle."""
        # Add a threat to trigger alerts
        await monitor.record_threat(sample_threat)

        # Perform monitoring check manually
        await monitor._perform_monitoring_check()

        # Should have updated last_check_time
        assert monitor.last_check_time is not None

        # Check should have processed threats
        assert len(monitor.active_threats) == 1

    @pytest.mark.asyncio
    async def test_threat_cleanup_in_monitoring(self, monitor):
        """Test that old threats are cleaned up during monitoring."""
        # Add old threat
        old_threat = ThreatDetection(
            threat_id="old_threat",
            threat_type=ThreatType.SQL_INJECTION,
            threat_level=ThreatLevel.HIGH,
            confidence=0.8,
            description="Old threat",
            source_events=[],
            mitigation_suggestions=[],
            timestamp=datetime.utcnow() - timedelta(hours=2)
        )
        monitor.active_threats.append(old_threat)

        # Add recent threat
        recent_threat = ThreatDetection(
            threat_id="recent_threat",
            threat_type=ThreatType.RATE_LIMIT_ABUSE,
            threat_level=ThreatLevel.MEDIUM,
            confidence=0.7,
            description="Recent threat",
            source_events=[],
            mitigation_suggestions=[],
            timestamp=datetime.utcnow()
        )
        monitor.active_threats.append(recent_threat)

        # Run monitoring check
        await monitor._perform_monitoring_check()

        # Old threat should be removed
        threat_ids = {t.threat_id for t in monitor.active_threats}
        assert "old_threat" not in threat_ids
        assert "recent_threat" in threat_ids

    @pytest.mark.asyncio
    async def test_get_security_dashboard_data(self, monitor, sample_event, sample_threat):
        """Test getting security dashboard data."""
        # Add some data
        await monitor.record_security_event(sample_event)
        await monitor.record_threat(sample_threat)

        dashboard_data = await monitor.get_security_dashboard_data()

        # Should contain all required sections
        assert 'timestamp' in dashboard_data
        assert 'monitoring_status' in dashboard_data
        assert 'metrics' in dashboard_data
        assert 'threats' in dashboard_data
        assert 'alerts' in dashboard_data
        assert 'recent_events' in dashboard_data

        # Monitoring status
        assert 'active' in dashboard_data['monitoring_status']
        assert 'last_check' in dashboard_data['monitoring_status']

        # Threat summary
        assert dashboard_data['threats']['total_active'] == 1
        assert dashboard_data['threats']['by_type']['SQL_INJECTION'] == 1

        # Recent events
        assert len(dashboard_data['recent_events']) >= 1

    def test_get_system_health(self, monitor):
        """Test system health status."""
        health = monitor.get_system_health()

        assert 'monitoring_active' in health
        assert 'last_check' in health
        assert 'active_threats' in health
        assert 'active_alerts' in health
        assert 'metric_types' in health
        assert 'alert_rules' in health
        assert 'event_buffer_size' in health

        # Should have default alert rules
        assert health['alert_rules'] > 0

    def test_default_alert_rules_setup(self, monitor):
        """Test that default alert rules are set up."""
        assert len(monitor.alerting.alert_rules) > 0

        # Check for expected default rules
        rule_names = {rule['name'] for rule in monitor.alerting.alert_rules}
        expected_rules = {'High Threat Count', 'Authentication Failures', 'Critical Threat Detected'}
        assert expected_rules.issubset(rule_names)

    @pytest.mark.asyncio
    async def test_monitoring_loop_error_handling(self, monitor):
        """Test error handling in monitoring loop."""
        # Mock an error in monitoring check
        original_check = monitor._perform_monitoring_check

        async def failing_check():
            raise Exception("Test error")

        monitor._perform_monitoring_check = failing_check

        # Start monitoring - should not crash
        await monitor.start_monitoring()

        # Give time for error handling
        await asyncio.sleep(0.1)

        # Monitoring should still be active
        assert monitor.monitoring_active

        # Restore original method and stop
        monitor._perform_monitoring_check = original_check
        await monitor.stop_monitoring()

    @pytest.mark.asyncio
    async def test_alert_generation_integration(self, monitor):
        """Test that monitoring generates alerts for threats."""
        # Add multiple high-level threats to trigger alert rules
        for i in range(6):
            threat = ThreatDetection(
                threat_id=f"alert_threat_{i}",
                threat_type=ThreatType.SQL_INJECTION,
                threat_level=ThreatLevel.HIGH,
                confidence=0.8,
                description=f"Test threat {i}",
                source_events=[],
                mitigation_suggestions=[]
            )
            await monitor.record_threat(threat)

        # Perform monitoring check
        await monitor._perform_monitoring_check()

        # Should have generated alerts
        active_alerts = monitor.alerting.get_active_alerts()
        assert len(active_alerts) > 0


class TestSecurityMonitorEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_empty_metrics_handling(self):
        """Test handling of empty metrics."""
        metrics = SecurityMetrics()

        # Get summary for empty metrics
        summary = await metrics.get_metric_summary(MetricType.THREAT_COUNT)

        assert summary['count'] == 0
        assert summary['trend'] == 'stable'

    @pytest.mark.asyncio
    async def test_malformed_alert_conditions(self):
        """Test handling of malformed alert conditions."""
        alerting = AlertingSystem()

        rule = {
            'name': 'Malformed Rule',
            'condition': 'completely invalid condition syntax',
            'alert_level': 'ERROR',
            'message': 'Test message'
        }

        alerting.add_alert_rule(rule)

        metrics = {'threat_count': {'total': 10}}

        # Should not crash with malformed condition
        alerts = await alerting.evaluate_alert_rules(metrics, [])
        assert isinstance(alerts, list)

    @pytest.mark.asyncio
    async def test_concurrent_monitoring_operations(self):
        """Test concurrent monitoring operations."""
        monitor = SecurityMonitor({
            'check_interval_seconds': 0.1,
            'metric_retention_hours': 1
        })

        # Simulate concurrent operations
        tasks = []

        # Start monitoring
        tasks.append(monitor.start_monitoring())

        # Record events concurrently
        for i in range(5):
            event = SecurityEvent(
                event_id=f"concurrent_{i}",
                timestamp=datetime.utcnow(),
                event_type="query",
                source_ip="192.168.1.100",
                user_id=f"user_{i}",
                collection="test_collection",
                query="SELECT * FROM documents"
            )
            tasks.append(monitor.record_security_event(event))

        # Wait for all operations
        await asyncio.gather(*tasks)

        # Give monitoring time to run
        await asyncio.sleep(0.2)

        # Stop monitoring
        await monitor.stop_monitoring()

        # Should have processed events without errors
        assert len(monitor.event_logger.event_buffer) >= 5

    @pytest.mark.asyncio
    async def test_memory_bounded_collections(self):
        """Test that collections are memory-bounded."""
        monitor = SecurityMonitor({
            'metric_retention_hours': 0.001  # Very short retention
        })

        # Add many metrics
        for i in range(1000):
            metric = SecurityMetric(
                metric_type=MetricType.THREAT_COUNT,
                value=i,
                timestamp=datetime.utcnow()
            )
            await monitor.metrics.record_metric(metric)

        # Collections should be bounded
        assert len(monitor.metrics.metrics[MetricType.THREAT_COUNT]) <= 10000
        assert len(monitor.event_logger.event_buffer) <= 10000

    def test_invalid_configuration_handling(self):
        """Test handling of invalid configurations."""
        # Should not crash with invalid config
        invalid_configs = [
            None,
            {},
            {'invalid_key': 'invalid_value'},
            {'check_interval_seconds': -1},
            {'metric_retention_hours': 'invalid'}
        ]

        for config in invalid_configs:
            try:
                monitor = SecurityMonitor(config)
                assert monitor is not None
            except Exception as e:
                pytest.fail(f"Failed to handle invalid config {config}: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
