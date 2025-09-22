"""
Unit tests for LSP Health Monitor

Tests cover:
- Health check functionality and status determination
- Automatic recovery mechanisms and strategies
- Fallback mode and graceful degradation
- User notification system
- Integration with AsyncioLspClient and PriorityQueueManager
- Error handling and edge cases
"""

import sys
from pathlib import Path

# Add src/python to path for common module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from typing import Dict, Any, List

from workspace_qdrant_mcp.core.lsp_health_monitor import (
    LspHealthMonitor,
    HealthCheckConfig,
    HealthStatus,
    RecoveryStrategy,
    NotificationLevel,
    HealthCheckResult,
    ServerHealthInfo,
    UserNotification,
)
from workspace_qdrant_mcp.core.lsp_client import (
    AsyncioLspClient,
    ConnectionState,
    CircuitBreakerState,
    LspError,
    LspTimeoutError,
    ServerCapabilities,
)
from workspace_qdrant_mcp.core.error_handling import ErrorCategory, ErrorSeverity


@pytest.fixture
def health_config():
    """Create test health check configuration"""
    return HealthCheckConfig(
        check_interval=1.0,
        fast_check_interval=0.5,
        health_check_timeout=2.0,
        consecutive_failures_threshold=2,
        recovery_success_threshold=2,
        max_recovery_attempts=3,
        base_backoff_delay=0.1,
        max_backoff_delay=1.0,
    )


@pytest.fixture
def mock_lsp_client():
    """Create mock LSP client"""
    client = Mock(spec=AsyncioLspClient)
    client.server_name = "test-server"
    client.is_connected.return_value = True
    client.is_circuit_open = False
    client.circuit_breaker_state = CircuitBreakerState.CLOSED
    client.communication_mode = Mock()
    client.communication_mode.name = "MANUAL"
    
    # Mock server capabilities
    capabilities = Mock(spec=ServerCapabilities)
    capabilities.supports_hover.return_value = True
    capabilities.supports_definition.return_value = True
    capabilities.supports_references.return_value = True
    capabilities.supports_document_symbol.return_value = True
    capabilities.supports_workspace_symbol.return_value = True
    client.server_capabilities = capabilities
    
    # Mock async methods
    client.workspace_symbol = AsyncMock(return_value=[])
    client.disconnect = AsyncMock()
    
    return client


@pytest.fixture
def mock_priority_queue():
    """Create mock priority queue manager"""
    queue = Mock()
    return queue


@pytest.fixture
def health_monitor(health_config, mock_priority_queue):
    """Create health monitor instance"""
    return LspHealthMonitor(
        config=health_config,
        priority_queue_manager=mock_priority_queue,
    )


class TestHealthCheckConfig:
    """Test health check configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = HealthCheckConfig()
        assert config.check_interval == 30.0
        assert config.health_check_timeout == 10.0
        assert config.consecutive_failures_threshold == 3
        assert config.enable_auto_recovery is True
        assert config.enable_fallback_mode is True
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = HealthCheckConfig(
            check_interval=15.0,
            max_recovery_attempts=10,
            enable_auto_recovery=False,
        )
        assert config.check_interval == 15.0
        assert config.max_recovery_attempts == 10
        assert config.enable_auto_recovery is False


class TestHealthCheckResult:
    """Test health check result handling"""
    
    def test_result_creation(self):
        """Test creating health check results"""
        timestamp = time.time()
        result = HealthCheckResult(
            timestamp=timestamp,
            status=HealthStatus.HEALTHY,
            response_time_ms=150.0,
            capabilities_valid=True,
        )
        
        assert result.timestamp == timestamp
        assert result.status == HealthStatus.HEALTHY
        assert result.response_time_ms == 150.0
        assert result.capabilities_valid is True
        assert result.error_message is None
    
    def test_result_with_error(self):
        """Test health check result with error information"""
        result = HealthCheckResult(
            timestamp=time.time(),
            status=HealthStatus.FAILED,
            response_time_ms=0.0,
            capabilities_valid=False,
            error_message="Connection failed",
            error_details={"code": 500, "retryable": True},
        )
        
        assert result.status == HealthStatus.FAILED
        assert result.error_message == "Connection failed"
        assert result.error_details["code"] == 500
    
    def test_result_to_dict(self):
        """Test converting result to dictionary"""
        result = HealthCheckResult(
            timestamp=123456789.0,
            status=HealthStatus.DEGRADED,
            response_time_ms=2500.0,
            capabilities_valid=False,
            error_message="Slow response",
        )
        
        data = result.to_dict()
        assert data["timestamp"] == 123456789.0
        assert data["status"] == "degraded"
        assert data["response_time_ms"] == 2500.0
        assert data["capabilities_valid"] is False
        assert data["error_message"] == "Slow response"


class TestServerHealthInfo:
    """Test server health information tracking"""
    
    def test_initial_health_info(self, mock_lsp_client):
        """Test initial server health info"""
        info = ServerHealthInfo(
            server_name="test-server",
            client=mock_lsp_client,
        )
        
        assert info.server_name == "test-server"
        assert info.client == mock_lsp_client
        assert info.current_status == HealthStatus.UNKNOWN
        assert info.is_healthy is False
        assert info.consecutive_failures == 0
        assert info.total_checks == 0
        assert info.uptime_percentage == 100.0
    
    def test_add_successful_check(self, mock_lsp_client):
        """Test adding successful health check"""
        info = ServerHealthInfo("test-server", mock_lsp_client)
        
        result = HealthCheckResult(
            timestamp=time.time(),
            status=HealthStatus.HEALTHY,
            response_time_ms=100.0,
        )
        
        info.add_check_result(result)
        
        assert info.current_status == HealthStatus.HEALTHY
        assert info.is_healthy is True
        assert info.consecutive_successes == 1
        assert info.consecutive_failures == 0
        assert info.total_checks == 1
        assert info.total_failures == 0
        assert info.uptime_percentage == 100.0
    
    def test_add_failed_check(self, mock_lsp_client):
        """Test adding failed health check"""
        info = ServerHealthInfo("test-server", mock_lsp_client)
        
        result = HealthCheckResult(
            timestamp=time.time(),
            status=HealthStatus.FAILED,
            response_time_ms=0.0,
            error_message="Connection failed",
        )
        
        info.add_check_result(result)
        
        assert info.current_status == HealthStatus.FAILED
        assert info.is_healthy is False
        assert info.consecutive_successes == 0
        assert info.consecutive_failures == 1
        assert info.total_checks == 1
        assert info.total_failures == 1
        assert info.uptime_percentage == 0.0
    
    def test_mixed_check_results(self, mock_lsp_client):
        """Test multiple check results with mixed outcomes"""
        info = ServerHealthInfo("test-server", mock_lsp_client)
        
        # Add successful check
        info.add_check_result(HealthCheckResult(
            timestamp=time.time(),
            status=HealthStatus.HEALTHY,
            response_time_ms=100.0,
        ))
        
        # Add failed check
        info.add_check_result(HealthCheckResult(
            timestamp=time.time(),
            status=HealthStatus.FAILED,
            response_time_ms=0.0,
        ))
        
        # Add another successful check
        info.add_check_result(HealthCheckResult(
            timestamp=time.time(),
            status=HealthStatus.HEALTHY,
            response_time_ms=150.0,
        ))
        
        assert info.current_status == HealthStatus.HEALTHY
        assert info.consecutive_successes == 1
        assert info.consecutive_failures == 0
        assert info.total_checks == 3
        assert info.total_failures == 1
        assert abs(info.uptime_percentage - 66.67) < 0.1
        assert abs(info.average_response_time - 83.33) < 0.1


class TestLspHealthMonitor:
    """Test LSP health monitor functionality"""
    
    def test_monitor_initialization(self, health_config, mock_priority_queue):
        """Test monitor initialization"""
        monitor = LspHealthMonitor(health_config, mock_priority_queue)
        
        assert monitor._config == health_config
        assert monitor._priority_queue_manager == mock_priority_queue
        assert len(monitor._servers) == 0
        assert len(monitor._monitoring_tasks) == 0
    
    def test_server_registration(self, health_monitor, mock_lsp_client):
        """Test registering LSP server for monitoring"""
        health_monitor.register_server("test-server", mock_lsp_client, auto_start_monitoring=False)
        
        assert "test-server" in health_monitor._servers
        assert health_monitor._servers["test-server"].server_name == "test-server"
        assert health_monitor._servers["test-server"].client == mock_lsp_client
        assert "test-server" in health_monitor._fallback_modes
        assert health_monitor._fallback_modes["test-server"] is False
    
    def test_server_unregistration(self, health_monitor, mock_lsp_client):
        """Test unregistering LSP server"""
        health_monitor.register_server("test-server", mock_lsp_client, auto_start_monitoring=False)
        health_monitor.unregister_server("test-server")
        
        assert "test-server" not in health_monitor._servers
        assert "test-server" not in health_monitor._fallback_modes
    
    def test_notification_handler_registration(self, health_monitor):
        """Test registering notification handlers"""
        handler = Mock()
        health_monitor.register_notification_handler(handler)
        
        assert handler in health_monitor._notification_handlers
    
    @pytest.mark.asyncio
    async def test_health_check_healthy_server(self, health_monitor, mock_lsp_client):
        """Test health check on healthy server"""
        health_monitor.register_server("test-server", mock_lsp_client, auto_start_monitoring=False)
        
        result = await health_monitor.perform_health_check("test-server")
        
        assert result.status == HealthStatus.HEALTHY
        assert result.response_time_ms > 0
        assert result.capabilities_valid is True
        assert result.error_message is None
        mock_lsp_client.workspace_symbol.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_disconnected_server(self, health_monitor, mock_lsp_client):
        """Test health check on disconnected server"""
        mock_lsp_client.is_connected.return_value = False
        health_monitor.register_server("test-server", mock_lsp_client, auto_start_monitoring=False)
        
        result = await health_monitor.perform_health_check("test-server")
        
        assert result.status == HealthStatus.DISCONNECTED
        assert result.capabilities_valid is False
        assert "not connected" in result.error_message
    
    @pytest.mark.asyncio
    async def test_health_check_circuit_breaker_open(self, health_monitor, mock_lsp_client):
        """Test health check with circuit breaker open"""
        mock_lsp_client.is_circuit_open = True
        health_monitor.register_server("test-server", mock_lsp_client, auto_start_monitoring=False)
        
        result = await health_monitor.perform_health_check("test-server")
        
        assert result.status == HealthStatus.FAILED
        assert result.capabilities_valid is False
        assert "circuit breaker" in result.error_message
    
    @pytest.mark.asyncio
    async def test_health_check_timeout(self, health_monitor, mock_lsp_client):
        """Test health check timeout"""
        mock_lsp_client.workspace_symbol.side_effect = asyncio.TimeoutError()
        health_monitor.register_server("test-server", mock_lsp_client, auto_start_monitoring=False)
        
        result = await health_monitor.perform_health_check("test-server")
        
        assert result.status == HealthStatus.UNHEALTHY
        assert result.capabilities_valid is False
        assert "timed out" in result.error_message
    
    @pytest.mark.asyncio
    async def test_health_check_lsp_error(self, health_monitor, mock_lsp_client):
        """Test health check with LSP error"""
        error = LspError("Server error", category=ErrorCategory.IPC, retryable=True)
        mock_lsp_client.workspace_symbol.side_effect = error
        health_monitor.register_server("test-server", mock_lsp_client, auto_start_monitoring=False)
        
        result = await health_monitor.perform_health_check("test-server")
        
        assert result.status == HealthStatus.UNHEALTHY
        assert result.capabilities_valid is False
        assert "Server error" in result.error_message
        assert result.error_details["retryable"] is True
    
    @pytest.mark.asyncio
    async def test_health_check_slow_response(self, health_monitor, mock_lsp_client):
        """Test health check with slow response (degraded status)"""
        async def slow_response(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate slow response
            return []
        
        mock_lsp_client.workspace_symbol.side_effect = slow_response
        
        # Configure for very fast degraded threshold
        with patch.object(health_monitor, '_config') as mock_config:
            mock_config.health_check_timeout = 10.0
            health_monitor.register_server("test-server", mock_lsp_client, auto_start_monitoring=False)
            
            result = await health_monitor.perform_health_check("test-server")
            
            # Should be healthy but slow
            assert result.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
            assert result.response_time_ms > 100  # At least 100ms due to sleep
    
    @pytest.mark.asyncio
    async def test_capability_validation(self, health_monitor, mock_lsp_client):
        """Test LSP capability validation"""
        health_monitor.register_server("test-server", mock_lsp_client, auto_start_monitoring=False)
        
        # Test with capabilities available
        result = await health_monitor.perform_health_check("test-server", include_capabilities=True)
        assert result.capabilities_valid is True
        
        # Test with no capabilities
        mock_lsp_client.server_capabilities = None
        result = await health_monitor.perform_health_check("test-server", include_capabilities=True)
        assert result.capabilities_valid is False
    
    def test_recovery_strategy_determination(self, health_monitor, mock_lsp_client):
        """Test recovery strategy determination"""
        health_monitor.register_server("test-server", mock_lsp_client, auto_start_monitoring=False)
        server_info = health_monitor._servers["test-server"]
        
        # Circuit breaker open -> circuit breaker recovery
        mock_lsp_client.is_circuit_open = True
        strategy = health_monitor._determine_recovery_strategy("test-server")
        assert strategy == RecoveryStrategy.CIRCUIT_BREAKER
        
        # Multiple failures -> backoff
        mock_lsp_client.is_circuit_open = False
        server_info.consecutive_failures = 5
        strategy = health_monitor._determine_recovery_strategy("test-server")
        assert strategy == RecoveryStrategy.BACKOFF
        
        # First attempt -> immediate
        server_info.consecutive_failures = 1
        server_info.recovery_attempts = 0
        strategy = health_monitor._determine_recovery_strategy("test-server")
        assert strategy == RecoveryStrategy.IMMEDIATE
    
    @pytest.mark.asyncio
    async def test_fallback_mode_activation(self, health_monitor, mock_lsp_client):
        """Test fallback mode activation"""
        health_monitor.register_server("test-server", mock_lsp_client, auto_start_monitoring=False)
        server_info = health_monitor._servers["test-server"]
        server_info.supported_features = {"hover", "definition", "references"}
        
        # Enable fallback mode
        result = await health_monitor._enable_fallback_mode("test-server")
        
        assert result is True
        assert health_monitor._fallback_modes["test-server"] is True
        assert "hover" in health_monitor._degraded_features
        assert "definition" in health_monitor._degraded_features
        assert "references" in health_monitor._degraded_features
    
    @pytest.mark.asyncio
    async def test_fallback_mode_disabled(self, health_monitor, mock_lsp_client):
        """Test fallback mode when disabled in config"""
        health_monitor._config.enable_fallback_mode = False
        health_monitor.register_server("test-server", mock_lsp_client, auto_start_monitoring=False)
        
        result = await health_monitor._enable_fallback_mode("test-server")
        
        assert result is False
        assert health_monitor._fallback_modes["test-server"] is False
    
    def test_feature_availability_check(self, health_monitor, mock_lsp_client):
        """Test checking feature availability"""
        health_monitor.register_server("test-server", mock_lsp_client, auto_start_monitoring=False)
        server_info = health_monitor._servers["test-server"]
        server_info.supported_features = {"hover", "definition", "references"}
        
        # Mark server as healthy
        result = HealthCheckResult(
            timestamp=time.time(),
            status=HealthStatus.HEALTHY,
            response_time_ms=100.0,
        )
        server_info.add_check_result(result)
        
        # Check available features
        assert health_monitor.is_feature_available("hover") is True
        assert health_monitor.is_feature_available("definition") is True
        assert health_monitor.is_feature_available("completion") is False
        
        # Degrade a feature
        health_monitor._degraded_features.add("hover")
        assert health_monitor.is_feature_available("hover") is False
    
    def test_get_available_features(self, health_monitor, mock_lsp_client):
        """Test getting all available features"""
        health_monitor.register_server("test-server", mock_lsp_client, auto_start_monitoring=False)
        server_info = health_monitor._servers["test-server"]
        server_info.supported_features = {"hover", "definition", "references"}
        
        # Mark server as healthy
        result = HealthCheckResult(
            timestamp=time.time(),
            status=HealthStatus.HEALTHY,
            response_time_ms=100.0,
        )
        server_info.add_check_result(result)
        
        features = health_monitor.get_available_features()
        assert "hover" in features
        assert "definition" in features
        assert "references" in features
        
        # Degrade some features
        health_monitor._degraded_features.update({"hover", "references"})
        features = health_monitor.get_available_features()
        assert "hover" not in features
        assert "definition" in features
        assert "references" not in features
    
    def test_health_statistics(self, health_monitor, mock_lsp_client):
        """Test getting health statistics"""
        health_monitor.register_server("test-server", mock_lsp_client, auto_start_monitoring=False)
        health_monitor._successful_recoveries = 5
        health_monitor._failed_recoveries = 2
        health_monitor._total_notifications_sent = 10
        
        stats = health_monitor.get_health_statistics()
        
        assert stats["total_servers"] == 1
        assert stats["successful_recoveries"] == 5
        assert stats["failed_recoveries"] == 2
        assert abs(stats["recovery_success_rate"] - (5/7)) < 0.01
        assert stats["total_notifications_sent"] == 10
        assert "test-server" in stats["servers"]
        assert stats["servers"]["test-server"]["status"] == "unknown"
    
    def test_troubleshooting_steps(self, health_monitor, mock_lsp_client):
        """Test troubleshooting step generation"""
        health_monitor.register_server("test-server", mock_lsp_client, auto_start_monitoring=False)
        
        # Test disconnected server
        result = HealthCheckResult(
            timestamp=time.time(),
            status=HealthStatus.DISCONNECTED,
            response_time_ms=0.0,
        )
        steps = health_monitor._get_troubleshooting_steps("test-server", result)
        assert any("process is running" in step for step in steps)
        assert any("startup command" in step for step in steps)
        
        # Test slow response
        result = HealthCheckResult(
            timestamp=time.time(),
            status=HealthStatus.HEALTHY,
            response_time_ms=2000.0,  # 2 seconds
        )
        steps = health_monitor._get_troubleshooting_steps("test-server", result)
        assert any("CPU and memory" in step for step in steps)
        
        # Test capability issues
        result = HealthCheckResult(
            timestamp=time.time(),
            status=HealthStatus.DEGRADED,
            response_time_ms=100.0,
            capabilities_valid=False,
        )
        steps = health_monitor._get_troubleshooting_steps("test-server", result)
        assert any("supports required LSP features" in step for step in steps)
    
    @pytest.mark.asyncio
    async def test_notification_sending(self, health_monitor):
        """Test sending user notifications"""
        handler = Mock()
        async_handler = AsyncMock()
        
        health_monitor.register_notification_handler(handler)
        health_monitor.register_notification_handler(async_handler)
        
        notification = UserNotification(
            timestamp=time.time(),
            level=NotificationLevel.WARNING,
            title="Test Notification",
            message="Test message",
            server_name="test-server",
        )
        
        await health_monitor._send_notification(notification)
        
        handler.assert_called_once_with(notification)
        async_handler.assert_called_once_with(notification)
        assert health_monitor._total_notifications_sent == 1
    
    @pytest.mark.asyncio
    async def test_notification_handler_error(self, health_monitor):
        """Test notification handler error handling"""
        def failing_handler(notification):
            raise Exception("Handler failed")
        
        health_monitor.register_notification_handler(failing_handler)
        
        notification = UserNotification(
            timestamp=time.time(),
            level=NotificationLevel.INFO,
            title="Test",
            message="Test",
            server_name="test-server",
        )
        
        # Should not raise exception
        await health_monitor._send_notification(notification)
        assert health_monitor._total_notifications_sent == 1
    
    @pytest.mark.asyncio
    async def test_notifications_disabled(self, health_monitor):
        """Test behavior when notifications are disabled"""
        health_monitor._config.enable_user_notifications = False
        handler = Mock()
        health_monitor.register_notification_handler(handler)
        
        notification = UserNotification(
            timestamp=time.time(),
            level=NotificationLevel.INFO,
            title="Test",
            message="Test",
            server_name="test-server",
        )
        
        await health_monitor._send_notification(notification)
        
        handler.assert_not_called()
        assert health_monitor._total_notifications_sent == 0
    
    @pytest.mark.asyncio
    async def test_monitoring_context(self, health_monitor, mock_lsp_client):
        """Test monitoring context manager"""
        health_monitor.register_server("test-server", mock_lsp_client, auto_start_monitoring=False)
        
        async with health_monitor.monitoring_context():
            # Should start monitoring
            assert "test-server" in health_monitor._monitoring_tasks
            task = health_monitor._monitoring_tasks["test-server"]
            assert not task.done()
        
        # Should stop monitoring after context
        # Note: In real test this would be verified, but we can't easily test the full loop
    
    @pytest.mark.asyncio 
    async def test_shutdown(self, health_monitor, mock_lsp_client):
        """Test monitor shutdown"""
        health_monitor.register_server("test-server", mock_lsp_client, auto_start_monitoring=False)
        
        # Create a real asyncio task that we can control
        async def dummy_task():
            try:
                await asyncio.sleep(10)  # Long sleep to simulate running task
            except asyncio.CancelledError:
                pass  # Expected when cancelled
        
        task = asyncio.create_task(dummy_task())
        health_monitor._monitoring_tasks["test-server"] = task
        
        await health_monitor.shutdown()
        
        assert task.cancelled() or task.done()
        assert len(health_monitor._monitoring_tasks) == 0


class TestUserNotification:
    """Test user notification functionality"""
    
    def test_notification_creation(self):
        """Test creating user notifications"""
        notification = UserNotification(
            timestamp=123456789.0,
            level=NotificationLevel.ERROR,
            title="Server Failed",
            message="LSP server has failed completely",
            server_name="python-lsp",
            troubleshooting_steps=["Check logs", "Restart server"],
            auto_recovery_attempted=True,
        )
        
        assert notification.timestamp == 123456789.0
        assert notification.level == NotificationLevel.ERROR
        assert notification.title == "Server Failed"
        assert notification.message == "LSP server has failed completely"
        assert notification.server_name == "python-lsp"
        assert notification.troubleshooting_steps == ["Check logs", "Restart server"]
        assert notification.auto_recovery_attempted is True
    
    def test_notification_to_dict(self):
        """Test converting notification to dictionary"""
        notification = UserNotification(
            timestamp=123456789.0,
            level=NotificationLevel.WARNING,
            title="Degraded Performance",
            message="Server is responding slowly",
            server_name="rust-analyzer",
            troubleshooting_steps=["Check CPU usage"],
        )
        
        data = notification.to_dict()
        assert data["timestamp"] == 123456789.0
        assert data["level"] == "warning"
        assert data["title"] == "Degraded Performance"
        assert data["message"] == "Server is responding slowly"
        assert data["server_name"] == "rust-analyzer"
        assert data["troubleshooting_steps"] == ["Check CPU usage"]
        assert data["auto_recovery_attempted"] is False


class TestIntegration:
    """Integration tests with real AsyncioLspClient behavior"""
    
    @pytest.mark.asyncio
    async def test_real_client_integration(self, health_config):
        """Test integration with real LSP client (mocked responses)"""
        # Create a more realistic mock that behaves like AsyncioLspClient
        client = Mock(spec=AsyncioLspClient)
        client.server_name = "test-lsp"
        client.is_connected.return_value = True
        client.is_circuit_open = False
        client.circuit_breaker_state = CircuitBreakerState.CLOSED
        
        # Mock capabilities
        capabilities = Mock(spec=ServerCapabilities)
        capabilities.supports_hover.return_value = True
        capabilities.supports_definition.return_value = True
        capabilities.supports_references.return_value = False
        capabilities.supports_document_symbol.return_value = True
        capabilities.supports_workspace_symbol.return_value = True
        client.server_capabilities = capabilities
        
        # Mock workspace_symbol to simulate real behavior
        client.workspace_symbol = AsyncMock()
        
        # Create monitor and register server
        monitor = LspHealthMonitor(config=health_config)
        monitor.register_server("test-lsp", client, auto_start_monitoring=False)
        
        # Perform health check
        result = await monitor.perform_health_check("test-lsp", include_capabilities=True)
        
        assert result.status == HealthStatus.HEALTHY
        assert result.capabilities_valid is True
        assert result.response_time_ms > 0
        
        # Verify capabilities were detected
        server_info = monitor.get_server_health("test-lsp")
        assert "hover" in server_info.supported_features
        assert "definition" in server_info.supported_features
        assert "references" not in server_info.supported_features  # Not supported
        
        # Mark server as healthy by adding the health check result
        server_info.add_check_result(result)
        
        # Check feature availability
        assert monitor.is_feature_available("hover") is True
        assert monitor.is_feature_available("references") is False
        
        client.workspace_symbol.assert_called_once_with("__health_check__")
    
    @pytest.mark.asyncio
    async def test_recovery_attempt_flow(self, health_config):
        """Test complete recovery attempt flow"""
        client = Mock(spec=AsyncioLspClient)
        client.server_name = "failing-server"
        client.is_connected.return_value = False  # Start disconnected
        client.is_circuit_open = False
        client.communication_mode = Mock()
        client.communication_mode.name = "TCP"
        client.disconnect = AsyncMock()
        
        monitor = LspHealthMonitor(config=health_config)
        notifications = []
        
        def capture_notification(notification: UserNotification):
            notifications.append(notification)
        
        monitor.register_notification_handler(capture_notification)
        monitor.register_server("failing-server", client, auto_start_monitoring=False)
        
        # Attempt recovery (should fail due to disconnected state)
        success = await monitor.attempt_recovery("failing-server")
        
        assert success is False
        assert len(notifications) == 0  # No immediate notification for failed recovery
        
        # Simulate reaching max recovery attempts by making additional attempts
        server_info = monitor.get_server_health("failing-server")
        
        # Make more recovery attempts to reach the maximum and trigger fallback
        # We already made 1 attempt above, need 2 more to reach max (3), then 1 more to trigger fallback
        for i in range(health_config.max_recovery_attempts):
            success = await monitor.attempt_recovery("failing-server")
            
            # The last attempt should trigger fallback mode
            if i == health_config.max_recovery_attempts - 1:
                # This attempt should have triggered fallback due to max attempts reached
                break
            else:
                assert success is False  # Should fail each time before fallback
        
        # Check that fallback mode was enabled after reaching max attempts
        assert monitor._fallback_modes["failing-server"] is True
        assert len(notifications) > 0  # Should have notifications about fallback mode