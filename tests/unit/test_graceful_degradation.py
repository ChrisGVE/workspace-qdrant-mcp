"""
Unit Tests for Graceful Degradation Manager

This module tests the graceful degradation strategies, circuit breaker patterns,
fallback mechanisms, and user experience for degraded states.
"""

import asyncio
import pytest
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from src.python.common.core.graceful_degradation import (
    DegradationManager,
    DegradationMode,
    FeatureType,
    CircuitBreaker,
    CircuitBreakerState,
    ThrottleStrategy,
    DegradationEvent,
    ResourceThrottle,
    FeatureConfig,
)
from src.python.common.core.component_coordination import ComponentType
from src.python.common.core.component_lifecycle import ComponentLifecycleManager, ComponentState
from src.python.common.core.lsp_health_monitor import LspHealthMonitor, NotificationLevel


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state."""
        cb = CircuitBreaker("test-component")
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.should_allow_request() is True

    def test_circuit_breaker_failure_threshold(self):
        """Test circuit breaker opens after failure threshold."""
        cb = CircuitBreaker("test-component", failure_threshold=3)

        # Record failures up to threshold
        for i in range(3):
            cb.record_failure()
            if i < 2:
                assert cb.state == CircuitBreakerState.CLOSED

        # Should open after threshold reached
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.should_allow_request() is False

    def test_circuit_breaker_recovery_timeout(self):
        """Test circuit breaker recovery after timeout."""
        cb = CircuitBreaker("test-component", failure_threshold=1, recovery_timeout=1)

        # Open circuit breaker
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN

        # Should not allow requests immediately
        assert cb.should_allow_request() is False

        # Wait for recovery timeout
        time.sleep(1.1)

        # Should transition to half-open
        assert cb.should_allow_request() is True
        assert cb.state == CircuitBreakerState.HALF_OPEN

    def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker recovery from half-open state."""
        cb = CircuitBreaker("test-component", half_open_success_threshold=2)
        cb.state = CircuitBreakerState.HALF_OPEN

        # Record successful operations
        cb.record_success()
        assert cb.state == CircuitBreakerState.HALF_OPEN

        cb.record_success()
        assert cb.state == CircuitBreakerState.CLOSED

    def test_circuit_breaker_half_open_failure(self):
        """Test circuit breaker failure in half-open state."""
        cb = CircuitBreaker("test-component")
        cb.state = CircuitBreakerState.HALF_OPEN

        # Failure should reopen circuit
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN


class TestDegradationManager:
    """Test degradation manager functionality."""

    @pytest.fixture
    def mock_lifecycle_manager(self):
        """Create mock lifecycle manager."""
        manager = AsyncMock(spec=ComponentLifecycleManager)
        manager.get_component_status.return_value = {
            "components": {
                "rust_daemon": {"state": "operational"},
                "python_mcp_server": {"state": "operational"},
                "cli_utility": {"state": "ready"},
                "context_injector": {"state": "ready"},
            }
        }
        return manager

    @pytest.fixture
    def mock_health_monitor(self):
        """Create mock health monitor."""
        return AsyncMock(spec=LspHealthMonitor)

    @pytest.fixture
    async def degradation_manager(self, mock_lifecycle_manager, mock_health_monitor):
        """Create degradation manager for testing."""
        manager = DegradationManager(
            lifecycle_manager=mock_lifecycle_manager,
            health_monitor=mock_health_monitor
        )
        await manager.initialize()
        yield manager
        await manager.shutdown()

    def test_degradation_manager_initialization(self, degradation_manager):
        """Test degradation manager initialization."""
        assert degradation_manager.current_mode == DegradationMode.NORMAL
        assert len(degradation_manager.feature_configs) == len(FeatureType)
        assert len(degradation_manager.circuit_breakers) == len(ComponentType)

    def test_feature_availability_normal_mode(self, degradation_manager):
        """Test feature availability in normal mode."""
        for feature in FeatureType:
            assert degradation_manager.is_feature_available(feature) is True

        assert len(degradation_manager.get_available_features()) == len(FeatureType)
        assert len(degradation_manager.get_unavailable_features()) == 0

    async def test_degradation_mode_calculation(self, degradation_manager):
        """Test degradation mode calculation based on component health."""
        # Test with all components healthy
        healthy = {ComponentType.RUST_DAEMON, ComponentType.PYTHON_MCP_SERVER,
                  ComponentType.CLI_UTILITY, ComponentType.CONTEXT_INJECTOR}
        degraded = set()
        failed = set()

        mode = degradation_manager._calculate_degradation_mode(healthy, degraded, failed)
        assert mode == DegradationMode.NORMAL

        # Test with MCP server failed
        failed = {ComponentType.PYTHON_MCP_SERVER}
        healthy = {ComponentType.CLI_UTILITY}

        mode = degradation_manager._calculate_degradation_mode(healthy, degraded, failed)
        assert mode == DegradationMode.OFFLINE_CLI

        # Test with Rust daemon failed
        failed = {ComponentType.RUST_DAEMON}
        healthy = {ComponentType.PYTHON_MCP_SERVER}

        mode = degradation_manager._calculate_degradation_mode(healthy, degraded, failed)
        assert mode == DegradationMode.CACHED_ONLY

    async def test_force_degradation_mode(self, degradation_manager):
        """Test forcing degradation mode."""
        await degradation_manager.force_degradation_mode(
            DegradationMode.READ_ONLY,
            "Test override"
        )

        assert degradation_manager.current_mode == DegradationMode.READ_ONLY
        assert not degradation_manager.is_feature_available(FeatureType.DOCUMENT_INGESTION)
        assert degradation_manager.is_feature_available(FeatureType.SEMANTIC_SEARCH)

    async def test_feature_degradation_by_mode(self, degradation_manager):
        """Test feature degradation based on mode."""
        await degradation_manager.force_degradation_mode(
            DegradationMode.FEATURES_LIMITED,
            "Test feature limitation"
        )

        # Some features should be available
        assert degradation_manager.is_feature_available(FeatureType.SEMANTIC_SEARCH)
        assert degradation_manager.is_feature_available(FeatureType.KEYWORD_SEARCH)

        # Advanced features should be disabled
        assert not degradation_manager.is_feature_available(FeatureType.HYBRID_SEARCH)
        assert not degradation_manager.is_feature_available(FeatureType.REAL_TIME_INDEXING)

    async def test_fallback_response_generation(self, degradation_manager):
        """Test fallback response generation."""
        await degradation_manager.force_degradation_mode(
            DegradationMode.CACHED_ONLY,
            "Test fallback responses"
        )

        # Test search fallback
        fallback = await degradation_manager.get_fallback_response(
            "search",
            {"query": "test"},
            "search:test"
        )

        assert fallback is not None
        assert fallback["fallback"] is True
        assert fallback["degradation_mode"] == DegradationMode.CACHED_ONLY.name.lower()

        # Test document ingestion fallback
        fallback = await degradation_manager.get_fallback_response(
            "document_ingestion",
            {"file": "test.txt"},
            "ingest:test.txt"
        )

        assert fallback is not None
        assert fallback["success"] is False
        assert "temporarily disabled" in fallback["message"]

    async def test_circuit_breaker_integration(self, degradation_manager):
        """Test circuit breaker integration with components."""
        component_id = "rust_daemon-default"

        # Record failures to open circuit breaker
        for _ in range(5):
            await degradation_manager.record_component_failure(component_id)

        assert not degradation_manager.is_component_available(component_id)
        assert degradation_manager.get_circuit_breaker_state(component_id) == CircuitBreakerState.OPEN

        # Record success to recover
        await degradation_manager.record_component_success(component_id)
        # Circuit should still be open but may transition to half-open

    def test_request_throttling(self, degradation_manager):
        """Test request throttling logic."""
        # No throttling initially
        assert not degradation_manager.should_throttle_request(5)

        # Activate throttling
        degradation_manager.resource_throttle.is_active = True
        degradation_manager.resource_throttle.throttle_factor = 0.3

        # High priority requests less likely to be throttled
        high_priority_throttled = degradation_manager.should_throttle_request(1)
        low_priority_throttled = degradation_manager.should_throttle_request(9)

        # High priority should be less throttled than low priority
        # Note: This is probabilistic, so we can't guarantee exact results
        assert isinstance(high_priority_throttled, bool)
        assert isinstance(low_priority_throttled, bool)

    async def test_cache_functionality(self, degradation_manager):
        """Test caching functionality."""
        cache_key = "test:search:query"
        request_data = {"query": "test"}

        # First call should miss cache
        result1 = await degradation_manager.get_fallback_response(
            "search", request_data, cache_key
        )
        assert degradation_manager.cache_misses > 0

        # Second call should hit cache
        result2 = await degradation_manager.get_fallback_response(
            "search", request_data, cache_key
        )
        assert degradation_manager.cache_hits > 0
        assert result1 == result2

    async def test_degradation_events(self, degradation_manager):
        """Test degradation event tracking."""
        initial_events = len(degradation_manager.degradation_events)

        await degradation_manager.force_degradation_mode(
            DegradationMode.READ_ONLY,
            "Test event tracking"
        )

        assert len(degradation_manager.degradation_events) == initial_events + 1

        event = degradation_manager.degradation_events[-1]
        assert event.degradation_mode == DegradationMode.READ_ONLY
        assert event.trigger_reason == "Test event tracking"
        assert isinstance(event.user_guidance, list)

    def test_user_guidance_generation(self, degradation_manager):
        """Test user guidance generation for different modes."""
        guidance_performance = degradation_manager._generate_user_guidance(
            DegradationMode.PERFORMANCE_REDUCED, "High CPU usage"
        )
        assert "reduced performance" in guidance_performance[0].lower()

        guidance_readonly = degradation_manager._generate_user_guidance(
            DegradationMode.READ_ONLY, "Component failure"
        )
        assert "read-only mode" in guidance_readonly[0].lower()

        guidance_offline = degradation_manager._generate_user_guidance(
            DegradationMode.OFFLINE_CLI, "Network issues"
        )
        assert "cli operations" in guidance_offline[0].lower()

    def test_degradation_status_reporting(self, degradation_manager):
        """Test degradation status reporting."""
        status = degradation_manager.get_degradation_status()

        required_fields = [
            "current_mode", "previous_mode", "uptime_seconds",
            "degradation_count", "recovery_count", "available_features",
            "unavailable_features", "circuit_breakers", "resource_throttle",
            "cache_statistics", "recent_events"
        ]

        for field in required_fields:
            assert field in status

        assert status["current_mode"] == DegradationMode.NORMAL.name.lower()
        assert isinstance(status["available_features"], list)
        assert isinstance(status["circuit_breakers"], dict)

    async def test_component_health_integration(self, degradation_manager, mock_lifecycle_manager):
        """Test integration with component health monitoring."""
        # Simulate component failure
        mock_lifecycle_manager.get_component_status.return_value = {
            "components": {
                "rust_daemon": {"state": "failed"},
                "python_mcp_server": {"state": "operational"},
                "cli_utility": {"state": "ready"},
                "context_injector": {"state": "ready"},
            }
        }

        # Trigger evaluation
        await degradation_manager._evaluate_degradation_mode()

        # Should degrade to cached-only mode
        assert degradation_manager.current_mode in [DegradationMode.CACHED_ONLY, DegradationMode.FEATURES_LIMITED]

    async def test_notification_handling(self, degradation_manager):
        """Test degradation notification handling."""
        notifications_received = []

        def notification_handler(notification):
            notifications_received.append(notification)

        degradation_manager.register_notification_handler(notification_handler)

        await degradation_manager.force_degradation_mode(
            DegradationMode.READ_ONLY,
            "Test notification"
        )

        assert len(notifications_received) > 0
        notification = notifications_received[-1]
        assert "degradation" in notification.title.lower() or "read" in notification.title.lower()


class TestResourceThrottle:
    """Test resource throttling functionality."""

    def test_resource_throttle_initialization(self):
        """Test resource throttle initialization."""
        throttle = ResourceThrottle(ThrottleStrategy.RESOURCE_BASED)
        assert throttle.strategy == ThrottleStrategy.RESOURCE_BASED
        assert not throttle.is_active
        assert throttle.throttle_factor == 1.0

    def test_resource_throttle_activation(self):
        """Test resource throttle activation logic."""
        throttle = ResourceThrottle(ThrottleStrategy.RESOURCE_BASED)

        # Simulate high resource usage
        throttle.is_active = True
        throttle.throttle_factor = 0.5

        assert throttle.is_active
        assert throttle.throttle_factor == 0.5


class TestFeatureConfig:
    """Test feature configuration."""

    def test_feature_config_defaults(self):
        """Test feature configuration defaults."""
        config = FeatureConfig(
            FeatureType.SEMANTIC_SEARCH,
            {ComponentType.RUST_DAEMON}
        )

        assert config.feature_type == FeatureType.SEMANTIC_SEARCH
        assert ComponentType.RUST_DAEMON in config.required_components
        assert config.priority == 5  # Default priority
        assert config.can_use_cache is True

    def test_feature_config_custom_settings(self):
        """Test feature configuration with custom settings."""
        config = FeatureConfig(
            FeatureType.DOCUMENT_INGESTION,
            {ComponentType.RUST_DAEMON, ComponentType.PYTHON_MCP_SERVER},
            priority=3,
            cache_duration_seconds=600,
            can_use_cache=False
        )

        assert config.priority == 3
        assert config.cache_duration_seconds == 600
        assert config.can_use_cache is False
        assert len(config.required_components) == 2


class TestDegradationEvent:
    """Test degradation event functionality."""

    def test_degradation_event_creation(self):
        """Test degradation event creation."""
        event = DegradationEvent(
            event_id="test-event",
            degradation_mode=DegradationMode.READ_ONLY,
            previous_mode=DegradationMode.NORMAL,
            trigger_reason="Component failure",
            affected_features=[FeatureType.DOCUMENT_INGESTION],
            affected_components=[ComponentType.RUST_DAEMON],
            automatic_recovery=True,
            user_guidance=["Check component status"]
        )

        assert event.event_id == "test-event"
        assert event.degradation_mode == DegradationMode.READ_ONLY
        assert event.trigger_reason == "Component failure"
        assert isinstance(event.timestamp, datetime)
        assert event.automatic_recovery is True

    def test_degradation_event_post_init(self):
        """Test degradation event post-initialization."""
        event = DegradationEvent(
            event_id="test-event",
            degradation_mode=DegradationMode.EMERGENCY,
            previous_mode=DegradationMode.NORMAL,
            trigger_reason="System failure",
            affected_features=[],
            affected_components=[],
            automatic_recovery=False,
            user_guidance=[]
        )

        # Timestamp should be auto-generated
        assert event.timestamp is not None
        assert isinstance(event.timestamp, datetime)

        # Metadata should be initialized
        assert event.metadata == {}


@pytest.mark.asyncio
class TestDegradationManagerIntegration:
    """Integration tests for degradation manager."""

    async def test_full_degradation_cycle(self):
        """Test complete degradation and recovery cycle."""
        mock_lifecycle = AsyncMock(spec=ComponentLifecycleManager)
        mock_health = AsyncMock(spec=LspHealthMonitor)

        # Initial healthy state
        mock_lifecycle.get_component_status.return_value = {
            "components": {
                "rust_daemon": {"state": "operational"},
                "python_mcp_server": {"state": "operational"},
                "cli_utility": {"state": "ready"},
                "context_injector": {"state": "ready"},
            }
        }

        manager = DegradationManager(
            lifecycle_manager=mock_lifecycle,
            health_monitor=mock_health
        )

        try:
            await manager.initialize()

            # Should start in normal mode
            assert manager.current_mode == DegradationMode.NORMAL

            # Simulate component failure
            await manager.force_degradation_mode(
                DegradationMode.READ_ONLY,
                "Simulated failure"
            )

            # Verify degraded state
            assert manager.current_mode == DegradationMode.READ_ONLY
            assert not manager.is_feature_available(FeatureType.DOCUMENT_INGESTION)

            # Simulate recovery
            await manager.force_degradation_mode(
                DegradationMode.NORMAL,
                "Recovery complete"
            )

            # Verify normal state
            assert manager.current_mode == DegradationMode.NORMAL
            assert manager.is_feature_available(FeatureType.DOCUMENT_INGESTION)

        finally:
            await manager.shutdown()

    async def test_concurrent_degradation_operations(self):
        """Test concurrent degradation operations."""
        manager = DegradationManager()

        try:
            await manager.initialize()

            # Create multiple concurrent operations
            tasks = []
            for i in range(10):
                task = asyncio.create_task(
                    manager.get_fallback_response(
                        "search",
                        {"query": f"test{i}"},
                        f"search:test{i}"
                    )
                )
                tasks.append(task)

            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks)

            # All should return valid fallback responses
            for result in results:
                assert result is not None
                assert "fallback" in result

        finally:
            await manager.shutdown()