"""Unit tests for error monitoring system."""

import asyncio
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from common.core.error_categorization import ErrorCategory, ErrorSeverity
from common.core.error_message_manager import ErrorMessage
from common.core.error_monitoring import (
    CloudWatchHook,
    ErrorMetricsCollector,
    HealthCheckManager,
    HealthStatus,
    LoggingHook,
    MonitoringHook,
    OverallHealthStatus,
    PrometheusHook,
    WebhookHook,
)


class TestHealthStatus:
    """Test HealthStatus enum."""

    def test_to_numeric(self):
        """Test conversion to numeric values."""
        assert HealthStatus.HEALTHY.to_numeric() == 0
        assert HealthStatus.DEGRADED.to_numeric() == 1
        assert HealthStatus.UNHEALTHY.to_numeric() == 2


class TestOverallHealthStatus:
    """Test OverallHealthStatus dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        status = OverallHealthStatus(
            status=HealthStatus.HEALTHY,
            checks={"check1": HealthStatus.HEALTHY, "check2": HealthStatus.DEGRADED},
            details={"count": 5}
        )

        result = status.to_dict()

        assert result["status"] == "healthy"
        assert result["checks"]["check1"] == "healthy"
        assert result["checks"]["check2"] == "degraded"
        assert result["details"]["count"] == 5
        assert "timestamp" in result


class TestMonitoringHook:
    """Test MonitoringHook abstract base class."""

    def test_cannot_instantiate_abc(self):
        """Test that MonitoringHook cannot be instantiated directly."""
        with pytest.raises(TypeError):
            MonitoringHook()

    def test_subclass_must_implement_methods(self):
        """Test that subclass must implement abstract methods."""
        class IncompleteHook(MonitoringHook):
            pass

        with pytest.raises(TypeError):
            IncompleteHook()


class TestLoggingHook:
    """Test LoggingHook implementation."""

    @pytest.mark.asyncio
    async def test_emit_error(self, sample_error_message):
        """Test emitting error to logs."""
        hook = LoggingHook()

        with patch("common.core.error_monitoring.logger") as mock_logger:
            await hook.emit_error(sample_error_message)
            mock_logger.info.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_metric(self):
        """Test emitting metric to logs."""
        hook = LoggingHook()

        with patch("common.core.error_monitoring.logger") as mock_logger:
            await hook.emit_metric("test_metric", 42.5, {"label": "value"})
            mock_logger.debug.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_health_status(self):
        """Test emitting health status to logs."""
        hook = LoggingHook()
        status = OverallHealthStatus(
            status=HealthStatus.DEGRADED,
            checks={"check1": HealthStatus.DEGRADED}
        )

        with patch("common.core.error_monitoring.logger") as mock_logger:
            await hook.emit_health_status(status)
            # Should log at WARNING level for degraded status
            assert mock_logger.log.called


class TestWebhookHook:
    """Test WebhookHook implementation."""

    @pytest.mark.asyncio
    async def test_emit_error_with_httpx(self, sample_error_message):
        """Test emitting error to webhook."""
        mock_httpx = MagicMock()
        mock_client = AsyncMock()
        mock_httpx.AsyncClient.return_value = mock_client

        hook = WebhookHook("https://example.com/webhook")

        with patch.dict('sys.modules', {'httpx': mock_httpx}):
            await hook.emit_error(sample_error_message)
            mock_client.post.assert_called_once()
            args, kwargs = mock_client.post.call_args
            assert args[0] == "https://example.com/webhook"
            assert "json" in kwargs
            assert kwargs["json"]["event"] == "error"

    @pytest.mark.asyncio
    async def test_emit_metric(self):
        """Test emitting metric to webhook."""
        mock_httpx = MagicMock()
        mock_client = AsyncMock()
        mock_httpx.AsyncClient.return_value = mock_client

        hook = WebhookHook("https://example.com/webhook")

        with patch.dict('sys.modules', {'httpx': mock_httpx}):
            await hook.emit_metric("test_metric", 99.9, {"env": "prod"})
            mock_client.post.assert_called_once()
            args, kwargs = mock_client.post.call_args
            payload = kwargs["json"]
            assert payload["event"] == "metric"
            assert payload["data"]["name"] == "test_metric"
            assert payload["data"]["value"] == 99.9
            assert payload["data"]["labels"]["env"] == "prod"

    @pytest.mark.asyncio
    async def test_emit_health_status(self):
        """Test emitting health status to webhook."""
        mock_httpx = MagicMock()
        mock_client = AsyncMock()
        mock_httpx.AsyncClient.return_value = mock_client

        hook = WebhookHook("https://example.com/webhook")
        status = OverallHealthStatus(status=HealthStatus.HEALTHY)

        with patch.dict('sys.modules', {'httpx': mock_httpx}):
            await hook.emit_health_status(status)
            mock_client.post.assert_called_once()
            args, kwargs = mock_client.post.call_args
            payload = kwargs["json"]
            assert payload["event"] == "health_status"
            assert payload["data"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing webhook hook."""
        mock_httpx = MagicMock()
        mock_client = AsyncMock()
        mock_httpx.AsyncClient.return_value = mock_client

        hook = WebhookHook("https://example.com/webhook")

        with patch.dict('sys.modules', {'httpx': mock_httpx}):
            await hook.emit_metric("test", 1.0)
            await hook.close()
            mock_client.aclose.assert_called_once()


class TestPrometheusHook:
    """Test PrometheusHook implementation."""

    @pytest.mark.asyncio
    async def test_init_with_prometheus_client(self):
        """Test initialization with prometheus_client available."""
        mock_prom = MagicMock()
        mock_prom.Counter = MagicMock()
        mock_prom.Gauge = MagicMock()
        mock_prom.Histogram = MagicMock()

        with patch.dict('sys.modules', {'prometheus_client': mock_prom}):
            hook = PrometheusHook()
            assert hook._available

    @pytest.mark.asyncio
    async def test_init_without_prometheus_client(self):
        """Test initialization without prometheus_client."""
        with patch.dict('sys.modules', {'prometheus_client': None}):
            with patch('builtins.__import__', side_effect=ImportError):
                hook = PrometheusHook()
                assert not hook._available

    @pytest.mark.asyncio
    async def test_emit_error_when_available(self, sample_error_message):
        """Test emitting error when Prometheus available."""
        mock_prom = MagicMock()
        mock_counter = MagicMock()
        mock_prom.Counter.return_value = mock_counter
        mock_prom.Gauge = MagicMock()
        mock_prom.Histogram = MagicMock()

        with patch.dict('sys.modules', {'prometheus_client': mock_prom}):
            hook = PrometheusHook()
            await hook.emit_error(sample_error_message)

            # Verify counter was incremented
            assert mock_counter.labels.called

    @pytest.mark.asyncio
    async def test_emit_metric_error_rate(self):
        """Test emitting error_rate metric."""
        mock_prom = MagicMock()
        mock_gauge = MagicMock()
        mock_prom.Counter = MagicMock()
        mock_prom.Gauge.return_value = mock_gauge
        mock_prom.Histogram = MagicMock()

        with patch.dict('sys.modules', {'prometheus_client': mock_prom}):
            hook = PrometheusHook()
            await hook.emit_metric("error_rate", 5.2)

            # Verify gauge was set
            assert mock_gauge.set.called


class TestCloudWatchHook:
    """Test CloudWatchHook implementation."""

    @pytest.mark.asyncio
    async def test_init_with_boto3(self):
        """Test initialization with boto3 available."""
        mock_boto3 = MagicMock()
        mock_boto3.client.return_value = MagicMock()

        with patch.dict('sys.modules', {'boto3': mock_boto3}):
            hook = CloudWatchHook()
            assert hook._available
            mock_boto3.client.assert_called_once_with('cloudwatch')

    @pytest.mark.asyncio
    async def test_init_without_boto3(self):
        """Test initialization without boto3."""
        with patch.dict('sys.modules', {'boto3': None}):
            with patch('builtins.__import__', side_effect=ImportError):
                hook = CloudWatchHook()
                assert not hook._available

    @pytest.mark.asyncio
    async def test_emit_error_when_available(self, sample_error_message):
        """Test emitting error to CloudWatch."""
        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        with patch.dict('sys.modules', {'boto3': mock_boto3}):
            hook = CloudWatchHook(namespace="TestNamespace")
            await hook.emit_error(sample_error_message)

            mock_client.put_metric_data.assert_called_once()
            call_args = mock_client.put_metric_data.call_args
            assert call_args[1]["Namespace"] == "TestNamespace"


class TestErrorMetricsCollector:
    """Test ErrorMetricsCollector."""

    def test_register_hook(self):
        """Test registering monitoring hooks."""
        collector = ErrorMetricsCollector()
        hook = LoggingHook()

        result = collector.register_hook(hook)

        assert result is True
        assert hook in collector.hooks

    def test_register_duplicate_hook(self):
        """Test registering same hook twice."""
        collector = ErrorMetricsCollector()
        hook = LoggingHook()

        collector.register_hook(hook)
        result = collector.register_hook(hook)

        assert result is False
        assert len(collector.hooks) == 1

    @pytest.mark.asyncio
    async def test_emit_error_metric(self, sample_error_message):
        """Test emitting error metric to hooks."""
        collector = ErrorMetricsCollector()
        mock_hook = AsyncMock(spec=MonitoringHook)
        collector.register_hook(mock_hook)

        await collector.emit_error_metric(sample_error_message)

        mock_hook.emit_error.assert_called_once_with(sample_error_message)
        # Check internal metrics updated
        key = f"{sample_error_message.severity.value}_{sample_error_message.category.value}"
        assert collector._metrics["error_total"][key] == 1

    @pytest.mark.asyncio
    async def test_emit_error_rate(self):
        """Test emitting error rate metric."""
        collector = ErrorMetricsCollector()
        mock_hook = AsyncMock(spec=MonitoringHook)
        collector.register_hook(mock_hook)

        await collector.emit_error_rate(5.2, "1m")

        mock_hook.emit_metric.assert_called_once_with(
            "error_rate", 5.2, {"window": "1m"}
        )
        assert collector._metrics["error_rate"] == 5.2

    @pytest.mark.asyncio
    async def test_emit_acknowledgment_metric(self):
        """Test emitting acknowledgment metric."""
        collector = ErrorMetricsCollector()
        mock_hook = AsyncMock(spec=MonitoringHook)
        collector.register_hook(mock_hook)

        await collector.emit_acknowledgment_metric(45.5)

        mock_hook.emit_metric.assert_called_once_with(
            "acknowledgment_time_seconds", 45.5
        )
        assert 45.5 in collector._metrics["acknowledgment_times"]

    def test_get_current_metrics(self):
        """Test getting current metrics snapshot."""
        collector = ErrorMetricsCollector()
        collector._metrics["error_total"]["error_file_corrupt"] = 5
        collector._metrics["error_rate"] = 3.2
        collector._metrics["acknowledgment_times"] = [10.0, 20.0, 30.0]

        metrics = collector.get_current_metrics()

        assert metrics["error_total"]["error_file_corrupt"] == 5
        assert metrics["error_rate"] == 3.2
        assert metrics["acknowledgment_times"]["count"] == 3
        assert metrics["acknowledgment_times"]["avg"] == 20.0
        assert metrics["acknowledgment_times"]["min"] == 10.0
        assert metrics["acknowledgment_times"]["max"] == 30.0

    @pytest.mark.asyncio
    async def test_hook_failure_handling(self, sample_error_message):
        """Test graceful handling of hook failures."""
        collector = ErrorMetricsCollector()
        failing_hook = AsyncMock(spec=MonitoringHook)
        failing_hook.emit_error.side_effect = Exception("Hook failed")
        good_hook = AsyncMock(spec=MonitoringHook)

        collector.register_hook(failing_hook)
        collector.register_hook(good_hook)

        # Should not raise exception
        await collector.emit_error_metric(sample_error_message)

        # Good hook should still be called
        good_hook.emit_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing collector and hooks."""
        collector = ErrorMetricsCollector()
        mock_hook = AsyncMock(spec=MonitoringHook)
        collector.register_hook(mock_hook)

        await collector.close()

        mock_hook.close.assert_called_once()


class TestHealthCheckManager:
    """Test HealthCheckManager."""

    @pytest.fixture
    async def health_manager(self, mock_error_manager):
        """Create health check manager with mock error manager."""
        manager = HealthCheckManager(mock_error_manager)
        await manager.initialize()
        return manager

    @pytest.mark.asyncio
    async def test_initialize(self, mock_error_manager):
        """Test initialization."""
        manager = HealthCheckManager(mock_error_manager)

        await manager.initialize()

        assert manager._initialized
        mock_error_manager.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_error_threshold_healthy(self, health_manager, mock_error_manager):
        """Test error threshold check - healthy status."""
        # Mock 5 errors (below threshold of 10)
        mock_error_manager.get_errors.return_value = [MagicMock()] * 5

        status = await health_manager.check_error_threshold("error", 10, 5)

        assert status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_check_error_threshold_degraded(self, health_manager, mock_error_manager):
        """Test error threshold check - degraded status."""
        # Mock 8 errors (80% of threshold)
        mock_error_manager.get_errors.return_value = [MagicMock()] * 8

        status = await health_manager.check_error_threshold("error", 10, 5)

        assert status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_check_error_threshold_unhealthy(self, health_manager, mock_error_manager):
        """Test error threshold check - unhealthy status."""
        # Mock 10 errors (at threshold)
        mock_error_manager.get_errors.return_value = [MagicMock()] * 10

        status = await health_manager.check_error_threshold("error", 10, 5)

        assert status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_check_acknowledgment_rate_healthy(self, health_manager, mock_error_manager):
        """Test acknowledgment rate check - healthy status."""
        # Mock 8 acknowledged out of 10 errors (80% rate)
        mock_errors = [MagicMock(acknowledged=True)] * 8 + [MagicMock(acknowledged=False)] * 2
        mock_error_manager.get_errors.return_value = mock_errors

        status = await health_manager.check_acknowledgment_rate(0.8, 60)

        assert status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_check_acknowledgment_rate_degraded(self, health_manager, mock_error_manager):
        """Test acknowledgment rate check - degraded status."""
        # Mock 6 acknowledged out of 10 errors (60% rate, 70% of min_rate)
        mock_errors = [MagicMock(acknowledged=True)] * 6 + [MagicMock(acknowledged=False)] * 4
        mock_error_manager.get_errors.return_value = mock_errors

        status = await health_manager.check_acknowledgment_rate(0.8, 60)

        assert status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_check_acknowledgment_rate_unhealthy(self, health_manager, mock_error_manager):
        """Test acknowledgment rate check - unhealthy status."""
        # Mock 2 acknowledged out of 10 errors (20% rate)
        mock_errors = [MagicMock(acknowledged=True)] * 2 + [MagicMock(acknowledged=False)] * 8
        mock_error_manager.get_errors.return_value = mock_errors

        status = await health_manager.check_acknowledgment_rate(0.8, 60)

        assert status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_check_acknowledgment_rate_no_errors(self, health_manager, mock_error_manager):
        """Test acknowledgment rate with no errors."""
        mock_error_manager.get_errors.return_value = []

        status = await health_manager.check_acknowledgment_rate(0.8, 60)

        assert status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_get_health_status(self, health_manager, mock_error_manager):
        """Test getting overall health status."""
        # Mock error threshold check (healthy)
        mock_error_manager.get_errors.side_effect = [
            [MagicMock()] * 5,  # Error threshold check
            [MagicMock(acknowledged=True)] * 8 + [MagicMock(acknowledged=False)] * 2  # Ack rate
        ]

        # Mock stats
        mock_stats = MagicMock()
        mock_stats.to_dict.return_value = {"total_count": 10}
        mock_error_manager.get_error_stats.return_value = mock_stats

        status = await health_manager.get_health_status()

        assert status.status == HealthStatus.HEALTHY
        assert "error_threshold" in status.checks
        assert "acknowledgment_rate" in status.checks
        assert "error_stats" in status.details

    @pytest.mark.asyncio
    async def test_get_health_status_mixed_checks(self, health_manager, mock_error_manager):
        """Test overall status with mixed check results."""
        # Mock one healthy, one degraded check
        mock_error_manager.get_errors.side_effect = [
            [MagicMock()] * 5,  # Healthy
            [MagicMock(acknowledged=True)] * 6 + [MagicMock(acknowledged=False)] * 4  # Degraded
        ]

        mock_stats = MagicMock()
        mock_stats.to_dict.return_value = {}
        mock_error_manager.get_error_stats.return_value = mock_stats

        status = await health_manager.get_health_status()

        # Overall should be degraded if any check is degraded
        assert status.status == HealthStatus.DEGRADED

    def test_register_health_check(self, mock_error_manager):
        """Test registering custom health check."""
        manager = HealthCheckManager(mock_error_manager)

        def custom_check() -> HealthStatus:
            return HealthStatus.HEALTHY

        result = manager.register_health_check("custom", custom_check)

        assert result is True
        assert "custom" in manager._custom_checks

    def test_register_duplicate_health_check(self, mock_error_manager):
        """Test registering same health check twice."""
        manager = HealthCheckManager(mock_error_manager)

        def custom_check() -> HealthStatus:
            return HealthStatus.HEALTHY

        manager.register_health_check("custom", custom_check)
        result = manager.register_health_check("custom", custom_check)

        assert result is False

    @pytest.mark.asyncio
    async def test_close(self, mock_error_manager):
        """Test closing health check manager."""
        manager = HealthCheckManager(mock_error_manager)
        await manager.initialize()
        await manager.close()

        assert not manager._initialized
        mock_error_manager.close.assert_called_once()


@pytest.fixture
def sample_error_message():
    """Create sample error message for testing."""
    return ErrorMessage(
        id=1,
        timestamp=datetime.now(timezone.utc),
        severity=ErrorSeverity.ERROR,
        category=ErrorCategory.FILE_CORRUPT,
        message="Test error message",
        context={"file_path": "/test/file.txt"},
        acknowledged=False,
        retry_count=0
    )


@pytest.fixture
def mock_error_manager():
    """Create mock error message manager."""
    manager = AsyncMock()
    manager.initialize = AsyncMock()
    manager.close = AsyncMock()
    manager.get_errors = AsyncMock()
    manager.get_error_stats = AsyncMock()
    return manager
